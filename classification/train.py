#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2025 Nobuo Tsukamoto
This software is released under the MIT License.
See the LICENSE file in the project root for more information.
"""
import functools
import time
from typing import Dict
import math

import input_pipeline
import input_pipeline_for_efficientnet

import jax
import jax.numpy as jnp
import ml_collections
import models
import optax
import tensorflow_datasets as tfds

from absl import logging
from clu import metric_writers, periodic_actions
from flax import jax_utils
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from jax import lax

from train_state import TrainStateWithBatchNorm, TrainStateWithoutBatchNorm
from optimizer import create_learning_rate_fn, create_optimizer
from model_ema import ema_v2
from utils import get_input_dtype
from checkpoint import create_checkpoint_manager, restore_checkpoint, save_checkpoint

""" Training Image classfication model.

    based on:
        https://github.com/google/flax/blob/main/examples/imagenet/models.py
"""


def create_model(*, model_cls, half_precision, num_classes, **kwargs):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == "tpu":
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init(key, jnp.ones(input_shape, model.dtype))
    params = variables["params"]
    batch_stats = variables["batch_stats"] if "batch_stats" in variables else None

    return params, batch_stats


def cross_entropy_loss(logits, labels, num_classes, label_smoothing=0.0):
    if len(labels.shape) > 1:
        one_hot_labels = labels
    else:
        one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)

        if label_smoothing > 0.0:
            labels = optax.smooth_labels(one_hot_labels, label_smoothing)

    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def compute_metrics(logits, labels, num_classes, label_smoothing=0.0):
    loss = cross_entropy_loss(logits, labels, num_classes, label_smoothing)
    if len(labels.shape) > 1:
        accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    else:
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


def train_step(
    state,
    batch,
    rng,
    learning_rate_fn,
    num_classes,
    label_smoothing=0.0,
    with_batchnorm=True,
    gradient_accumulation_steps=1,
):
    """Perform a single training step."""

    dropout_rng, stochastic_depth_rng, next_rng = jax.random.split(rng, 3)

    def loss_fn(params):
        """loss function used for training."""
        if with_batchnorm:
            logits, new_model_state = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                batch["image"],
                mutable=["batch_stats"],
                rngs={"dropout": dropout_rng, "stochastic_depth": stochastic_depth_rng},
            )

        else:
            logits = state.apply_fn(
                {"params": params},
                batch["image"],
                rngs={"dropout": dropout_rng, "stochastic_depth": stochastic_depth_rng},
            )

        loss = cross_entropy_loss(logits, batch["label"], num_classes, label_smoothing)

        if with_batchnorm:
            return loss, (new_model_state, logits)
        else:
            return loss, logits

    step = state.step
    dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step // gradient_accumulation_steps)

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name="batch")
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = lax.pmean(grads, axis_name="batch")

    if with_batchnorm:
        new_model_state, logits = aux[1]
    else:
        logits = aux[1]

    metrics = compute_metrics(logits, batch["label"], num_classes, label_smoothing)
    metrics["learning_rate"] = lr

    if with_batchnorm:
        new_state = state.apply_gradients(
            grads=grads, batch_stats=new_model_state["batch_stats"]
        )
    else:
        new_state = state.apply_gradients(grads=grads)

    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        new_state = new_state.replace(
            opt_state=jax.tree_util.tree_map(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state,
            ),
            params=jax.tree_util.tree_map(
                functools.partial(jnp.where, is_fin), new_state.params, state.params
            ),
            dynamic_scale=dynamic_scale,
        )
        metrics["scale"] = dynamic_scale.scale

    if state.ema_tx is not None:
        _, new_ema_state = new_state.ema_tx.update(
            new_state.params, new_state.ema_state
        )
        new_state = new_state.replace(ema_state=new_ema_state)

        if with_batchnorm and new_state.ema_batch_stats is not None:
            _, new_ema_batch_stats = new_state.ema_tx.update(
                new_state.batch_stats, new_state.ema_batch_stats
            )

            new_state = new_state.replace(ema_batch_stats=new_ema_batch_stats)
    return new_state, metrics, next_rng


def eval_step(
    state, batch, num_classes, with_batchnorm, label_smoothing, model_ema=False
):
    if model_ema:
        params = state.ema_state.ema
        if state.ema_batch_stats is not None:
            batch_stats = state.ema_batch_stats.ema
        else:
            batch_stats = state.batch_stats
    else:
        params = state.params
        batch_stats = state.batch_stats

    if with_batchnorm:
        variables = {"params": params, "batch_stats": batch_stats}
        logits = state.apply_fn(variables, batch["image"], train=False, mutable=False)
    else:
        variables = {"params": params}
        logits = state.apply_fn(variables, batch["image"], train=False)

    return compute_metrics(
        logits, batch["label"], num_classes, label_smoothing=label_smoothing
    )


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, batch_size, dtype, train, config):
    if config.input_pipeline_type == "efficientnet":
        ds = input_pipeline_for_efficientnet.create_split(
            dataset_builder,
            batch_size=batch_size,
            dtype=dtype,
            train=train,
            config=config,
        )
    else:
        ds = input_pipeline.create_split(
            dataset_builder,
            batch_size=batch_size,
            dtype=dtype,
            train=train,
            config=config,
        )

    it = map(prepare_tf_data, ds)
    it = jax_utils.prefetch_to_device(it, 2)
    return it


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def create_train_state(
    rngs: Dict[str, jnp.ndarray],
    config: ml_collections.ConfigDict,
    model,
    learning_rate_fn,
):
    """Create initial training state."""
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == "gpu":
        dynamic_scale = dynamic_scale_lib.DynamicScale()
    else:
        dynamic_scale = None

    params, batch_stats = initialized(rngs, config.image_size, model)
    tx = create_optimizer(config, learning_rate_fn)

    ema_tx = None
    ema_state = None
    ema_batch_stats = None
    if config.model_ema:
        logging.info(
            "Model EMA %s, Decay rate: %f, Trainable weights only: %s",
            config.model_ema_type,
            config.model_ema_decay,
            config.model_ema_trainable_weights_only,
        )

        if config.model_ema_type == "v1":
            ema_tx = optax.ema(config.model_ema_decay)
        elif config.model_ema_type == "v2":
            ema_tx = ema_v2(config.model_ema_decay)
        else:
            logging.error("model_ema_type is incorrect")

        if config.gradient_accumulation_steps > 1:
            ema_tx = optax.MultiSteps(ema_tx, config.gradient_accumulation_steps)

        ema_state = ema_tx.init(params)

        if not config.model_ema_trainable_weights_only:
            ema_batch_stats = ema_tx.init(batch_stats)

    if batch_stats is not None:
        state = TrainStateWithBatchNorm.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            batch_stats=batch_stats,
            dynamic_scale=dynamic_scale,
            ema_tx=ema_tx,
            ema_state=ema_state,
            ema_batch_stats=ema_batch_stats,
        )
        with_batchnorm = True
    else:
        logging.info("Batch statistics are not found.")
        state = TrainStateWithoutBatchNorm.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            dynamic_scale=dynamic_scale,
            ema_tx=ema_tx,
            ema_state=ema_state,
        )
        with_batchnorm = False

    return state, with_batchnorm


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Execute model training and evaluation loop.
    Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the tensorboard summaries are written to.
    """

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )

    rng = jax.random.key(seed=config.seed)
    params_rng, dropout_rng, stochastic_depth_rng, train_rng = jax.random.split(
        rng, num=4
    )
    rngs = {
        "params": params_rng,
        "dropout": dropout_rng,
        "stochastic_depth": stochastic_depth_rng,
    }

    if config.batch_size % jax.device_count() > 0:
        raise ValueError("Batch size must be divisible by the number of devices")

    local_batch_size = config.batch_size // jax.process_count()
    input_dtype = get_input_dtype(config.half_precision)

    dataset_builder = tfds.builder(config.dataset, data_dir=config.dataset_dir)
    dataset_builder.download_and_prepare()
    train_iter = create_input_iter(
        dataset_builder, local_batch_size, input_dtype, train=True, config=config
    )
    eval_iter = create_input_iter(
        dataset_builder, local_batch_size, input_dtype, train=False, config=config
    )
    num_classes = dataset_builder.info.features["label"].num_classes

    steps_per_epoch = math.ceil(
        dataset_builder.info.splits["train"].num_examples
        / (config.batch_size * config.gradient_accumulation_steps)
    )
    steps_per_epoch *= config.gradient_accumulation_steps

    if config.num_train_steps <= 0:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps

    if config.steps_per_eval == -1:
        num_validation_examples = dataset_builder.info.splits["validation"].num_examples
        steps_per_eval = math.ceil(num_validation_examples / config.batch_size)
    else:
        steps_per_eval = config.steps_per_eval

    logging.info(
        "Steps per epech: %d, Step per eval: %d.", steps_per_epoch, steps_per_eval
    )

    steps_per_checkpoint = steps_per_epoch
    model_cls = getattr(models, config.model)
    model = create_model(
        model_cls=model_cls,
        half_precision=config.half_precision,
        num_classes=num_classes,
        init_stochastic_depth_rate=config.init_stochastic_depth_rate,
    )

    learning_rate_fn = create_learning_rate_fn(config, steps_per_epoch)
    state, with_batchnorm = create_train_state(rngs, config, model, learning_rate_fn)

    checkpoint_manager = create_checkpoint_manager(workdir, config)
    state = restore_checkpoint(checkpoint_manager, state)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            learning_rate_fn=learning_rate_fn,
            num_classes=num_classes,
            label_smoothing=config.label_smoothing,
            with_batchnorm=with_batchnorm,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        ),
        axis_name="batch",
    )
    p_eval_step = jax.pmap(
        functools.partial(
            eval_step,
            num_classes=num_classes,
            with_batchnorm=with_batchnorm,
            label_smoothing=config.label_smoothing,
        ),
        axis_name="batch",
    )
    p_ema_eval_step = jax.pmap(
        functools.partial(
            eval_step,
            num_classes=num_classes,
            with_batchnorm=with_batchnorm,
            label_smoothing=config.label_smoothing,
            model_ema=True,
        ),
        axis_name="batch",
    )
    train_rng = jax.random.split(train_rng, jax.local_device_count())
    train_metrics = []
    hooks = []
    if jax.process_index() == 0 and config.profile:
        hooks += [
            periodic_actions.Profile(
                num_profile_steps=3, profile_duration_ms=None, logdir=workdir
            )
        ]
    train_metrics_last_t = time.time()
    logging.info("Initial compilation, this might take some minutes...")

    for step, batch in zip(range(step_offset, num_steps), train_iter):
        state, metrics, train_rng = p_train_step(
            state,
            batch,
            train_rng,
        )
        for h in hooks:
            h(step)
        if step == step_offset:
            logging.info("Initial compilation completed.")

        if config.get("log_every_steps"):
            train_metrics.append(metrics)
            if (step + 1) % config.log_every_steps == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f"train_{k}": v
                    for k, v in jax.tree_util.tree_map(
                        lambda x: x.mean(), train_metrics
                    ).items()
                }
                summary["steps_per_second"] = config.log_every_steps / (
                    time.time() - train_metrics_last_t
                )
                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

        if (step + 1) % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            eval_metrics = []

            if with_batchnorm and config.use_sync_batch_norm:
                # Sync batch statistics across replicas before evaluation or
                # checkpointing.
                state = sync_batch_stats(state)

            for _ in range(steps_per_eval):
                eval_batch = next(eval_iter)
                metrics = p_eval_step(state, eval_batch)
                eval_metrics.append(metrics)
            eval_metrics = common_utils.get_metrics(eval_metrics)
            summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
            logging.info(
                "eval epoch: %d, loss: %.4f, accuracy: %.2f",
                epoch,
                summary["loss"],
                summary["accuracy"] * 100,
            )
            writer.write_scalars(
                step + 1, {f"eval_{key}": val for key, val in summary.items()}
            )
            writer.flush()

            if state.ema_tx is not None:
                eval_ema_metrics = []
                for _ in range(steps_per_eval):
                    eval_batch = next(eval_iter)
                    metrics = p_ema_eval_step(state, eval_batch)
                    eval_ema_metrics.append(metrics)
                eval_ema_metrics = common_utils.get_metrics(eval_ema_metrics)
                ema_summary = jax.tree_util.tree_map(
                    lambda x: x.mean(), eval_ema_metrics
                )
                logging.info(
                    "eval epoch: %d, loss: %.4f, accuracy: %.2f",
                    epoch,
                    ema_summary["loss"],
                    ema_summary["accuracy"] * 100,
                )
                writer.write_scalars(
                    step + 1,
                    {f"eval_ema_{key}": val for key, val in ema_summary.items()},
                )
                writer.flush()

        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            if with_batchnorm and not config.use_sync_batch_norm:
                # Final synchronization of batch statistics before checkpointing.
                # If SyncBN was not used during training, batch_stats may differ across
                # devices.
                # We synchronize here to ensure the saved model has consistent stats
                # for evaluation and inference.
                state = sync_batch_stats(state)
            save_checkpoint(checkpoint_manager, state)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(seed=config.seed), ()).block_until_ready()

    return state
