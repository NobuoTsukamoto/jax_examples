#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import functools
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

from absl import logging
from clu import metric_writers, periodic_actions
from flax import jax_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import checkpoints, common_utils, train_state
from jax import lax

import input_pipeline
import models
from miou_metrics import eval_semantic_segmentation
from loss import cross_entropy_loss, ohem_cross_entropy_loss, recall_cross_entory_loss


def create_model(*, model_cls, half_precision, num_classes, output_size, **kwargs):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == "tpu":
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_cls(
        num_classes=num_classes, output_size=output_size, dtype=model_dtype, **kwargs
    )


def initialized(key, image_size, model):
    input_shape = (1, image_size[0], image_size[1], 3)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init(key, jnp.ones(input_shape, model.dtype))
    return variables["params"], variables["batch_stats"]


def semantic_segmentation_metrics(logits, labels, num_classes, ignore_label):
    return eval_semantic_segmentation(
        jnp.argmax(logits, axis=-1),
        jnp.squeeze(labels, axis=-1),
        num_classes,
        ignore_label,
    )


def compute_metrics(logits, labels, num_classes, ignore_label, class_weights=None):
    loss = recall_cross_entory_loss(
        logits,
        labels,
        num_classes,
        ignore_label=ignore_label,
        class_weights=class_weights,
    )
    segmentation_metrics = semantic_segmentation_metrics(
        logits, labels, num_classes, ignore_label
    )
    metrics = {
        "miou": segmentation_metrics["miou"],
        "pixel_accuracy": segmentation_metrics["pixel_accuracy"],
        "mean_class_accuracy": segmentation_metrics["mean_class_accuracy"],
        "loss": loss,
    }
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


def create_learning_rate_fn(
    config: ml_collections.ConfigDict, base_learning_rate: float, steps_per_epoch: int
):
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


def train_step(
    state,
    batch,
    learning_rate_fn,
    num_classes,
    ignore_label,
    class_weights=None,
    dropout_rng=None,
):
    """Perform a single training step."""

    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        """loss function used for training."""
        logits, new_model_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch["image"],
            mutable=["batch_stats"],
            rngs={"dropout": dropout_rng},
        )
        loss = recall_cross_entory_loss(
            logits["output"],
            batch["label"],
            num_classes,
            ignore_label=ignore_label,
            class_weights=class_weights,
        )
        aux_loss = 0
        if "aux_loss" in logits:
            aux_loss = 0.4 * recall_cross_entory_loss(
                logits["aux_loss"],
                batch["label"],
                num_classes,
                ignore_label=ignore_label,
                class_weights=class_weights,
            )
        weight_decay = 0.00004
        # weight_penalty_params = jax.tree_util.tree_leaves(params)
        # weight_l2 = sum([jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
        weight_penalty_params = jax.tree_util.tree_leaves_with_path(params)
        weight_l2 = sum(
            jnp.sum(x[1] ** 2)
            for x in weight_penalty_params
            if x[1].ndim > 1 and "DepthwiseSeparable" not in x[0][0].key
        )
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + aux_loss + weight_penalty
        return loss, (new_model_state, logits)

    step = state.step
    dynamic_scale = state.dynamic_scale
    if learning_rate_fn is not None:
        lr = learning_rate_fn(step)

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True, axis_name="batch")
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = lax.pmean(grads, axis_name="batch")
    new_model_state, logits = aux[1]
    metrics = compute_metrics(
        logits["output"], batch["label"], num_classes, ignore_label, class_weights
    )

    if learning_rate_fn is not None:
        metrics["learning_rate"] = lr

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )
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

    return new_state, metrics, new_dropout_rng


def eval_step(state, batch, num_classes, ignore_label, class_weights=None):
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, batch["image"], train=False, mutable=False)
    return compute_metrics(
        logits["output"], batch["label"], num_classes, ignore_label, class_weights
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


def create_input_iter(
    dataset_builder,
    batch_size,
    input_image_size,
    crop_image_size,
    min_resize_value,
    max_resize_value,
    output_image_size,
    dtype,
    train,
    cache,
    ignore_label,
):
    ds = input_pipeline.create_split(
        dataset_builder,
        batch_size,
        input_image_size=input_image_size,
        crop_image_size=crop_image_size,
        min_resize_value=min_resize_value,
        max_resize_value=max_resize_value,
        output_image_size=output_image_size,
        dtype=dtype,
        train=train,
        cache=cache,
        ignore_label=ignore_label,
    )
    it = map(prepare_tf_data, ds)
    it = jax_utils.prefetch_to_device(it, 2)
    return it


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def get_input_dtype(half_precision):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == "tpu":
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else:
        input_dtype = tf.float32

    return input_dtype


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


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
    image_size,
    learning_rate_fn,
):
    """Create initial training state."""
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == "gpu":
        dynamic_scale = dynamic_scale_lib.DynamicScale()
    else:
        dynamic_scale = None

    params, batch_stats = initialized(rngs, image_size, model)
    if config.optimizer == "adamw":
        tx = optax.adamw(
            learning_rate=learning_rate_fn,
        )

    elif config.optimizer == "sgd":
        tx = optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=config.momentum,
            nesterov=True,
        )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        dynamic_scale=dynamic_scale,
    )
    return state


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    """Execute model training and evaluation loop.
    Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the tensorboard summaries are written to.
    Returns:
        Final TrainState.
    """

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )

    rng = jax.random.PRNGKey(0)
    params_rng, dropout_rng = jax.random.split(rng)
    rngs = {
        "params": params_rng,
        "dropout": dropout_rng,
    }

    if config.batch_size % jax.device_count() > 0:
        raise ValueError("Batch size must be divisible by the number of devices")

    local_batch_size = config.batch_size // jax.process_count()
    input_dtype = get_input_dtype(config.half_precision)

    dataset_builder = tfds.builder(config.dataset, data_dir=config.dataset_dir)
    dataset_builder.download_and_prepare()
    train_iter = create_input_iter(
        dataset_builder,
        local_batch_size,
        config.image_size,
        config.crop_image_size,
        config.min_resize_value,
        config.max_resize_value,
        config.output_image_size,
        input_dtype,
        train=True,
        cache=config.cache,
        ignore_label=config.ignore_label,
    )
    eval_iter = create_input_iter(
        dataset_builder,
        local_batch_size,
        config.image_size,
        config.crop_image_size,
        config.min_resize_value,
        config.max_resize_value,
        config.output_image_size,
        input_dtype,
        train=False,
        cache=config.cache,
        ignore_label=config.ignore_label,
    )
    num_classes = config.num_classes
    output_size = config.output_image_size

    steps_per_epoch = (
        dataset_builder.info.splits["train"].num_examples // config.batch_size
    )

    if config.num_train_steps == -1:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps

    if config.steps_per_eval == -1:
        num_validation_examples = dataset_builder.info.splits["validation"].num_examples
        steps_per_eval = num_validation_examples // config.batch_size
    else:
        steps_per_eval = config.steps_per_eval

    steps_per_checkpoint = steps_per_epoch
    model_cls = getattr(models, config.model)
    model = create_model(
        model_cls=model_cls,
        half_precision=config.half_precision,
        num_classes=num_classes,
        output_size=output_size,
    )

    base_learning_rate = config.learning_rate
    learning_rate_fn = create_learning_rate_fn(
        config, base_learning_rate, steps_per_epoch
    )

    state = create_train_state(rngs, config, model, config.image_size, learning_rate_fn)
    state = restore_checkpoint(state, workdir)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            learning_rate_fn=learning_rate_fn,
            num_classes=num_classes,
            ignore_label=config.ignore_label,
            class_weights=config.class_weights,
        ),
        axis_name="batch",
    )
    p_eval_step = jax.pmap(
        functools.partial(
            eval_step,
            num_classes=num_classes,
            ignore_label=config.ignore_label,
            class_weights=config.class_weights,
        ),
        axis_name="batch",
    )

    dropout_rngs = jax.random.split(dropout_rng, jax.local_device_count())
    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()
    logging.info("Initial compilation, this might take some minutes...")

    for step, batch in zip(range(step_offset, num_steps), train_iter):
        state, metrics, dropout_rngs = p_train_step(
            state, batch, dropout_rng=dropout_rngs
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

            # sync batch statistics across replicas
            state = sync_batch_stats(state)
            for i in range(steps_per_eval):
                eval_batch = next(eval_iter)
                metrics = p_eval_step(state, eval_batch)
                eval_metrics.append(metrics)

            eval_metrics = common_utils.get_metrics(eval_metrics)
            summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
            logging.info(
                "eval epoch: %d, loss: %.4f, miou: %.4f, class accuracy: %.4f, pixel accuracy: %.4f",
                epoch,
                summary["loss"],
                summary["miou"],
                summary["mean_class_accuracy"],
                summary["pixel_accuracy"],
            )
            writer.write_scalars(
                step + 1, {f"eval_{key}": val for key, val in summary.items()}
            )
            writer.flush()

        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            state = sync_batch_stats(state)
            save_checkpoint(state, workdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state
