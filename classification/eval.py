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

import jax
import jax.numpy as jnp
import ml_collections
import models
import optax
import tensorflow_datasets as tfds
from absl import logging
from flax import jax_utils
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from jax import lax

from train import create_model, initialized, create_input_iter
from train_state import TrainStateWithBatchNorm, TrainStateWithoutBatchNorm
from optimizer import create_learning_rate_fn, create_optimizer
from utils import get_input_dtype
from loss import cross_entropy_loss
from checkpoint import create_checkpoint_manager, restore_checkpoint
from model_ema import ema_v2

""" Eval Image classfication model.

    based on:
        https://github.com/google/flax/blob/main/examples/imagenet/models.py
"""


def compute_metrics(logits, labels, num_classes):
    loss = cross_entropy_loss(logits, labels, num_classes)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    top_5_indices = jax.lax.top_k(logits, k=5)[1]
    correct_in_top_5 = jnp.any(top_5_indices == labels[:, None], axis=1)
    top_5_accuracy = jnp.mean(correct_in_top_5)

    metrics = {
        "loss": loss,
        "top-1 accuracy": accuracy,
        "top-5 accuracy": top_5_accuracy,
    }
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


def eval_step(state, batch, num_classes, with_batchnorm, model_ema=False):
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

    return compute_metrics(logits, batch["label"], num_classes)


def create_eval_state(
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
        if config.model_ema_type == "v1":
            ema_tx = optax.ema(config.model_ema_decay)
        elif config.model_ema_type == "v2":
            ema_tx = ema_v2(config.model_ema_decay)
        else:
            logging.error("model_ema_type is incorrect")

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


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Execute model training and evaluation loop.
    Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the tensorboard summaries are written to.
    """

    rng = jax.random.PRNGKey(seed=config.seed)
    params_rng, dropout_rng, stochastic_depth_rng = jax.random.split(rng, num=3)
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
    steps_per_epoch = math.ceil(
        dataset_builder.info.splits["train"].num_examples
        / (config.batch_size * config.gradient_accumulation_steps)
    )
    steps_per_epoch *= config.gradient_accumulation_steps

    eval_iter = create_input_iter(
        dataset_builder, local_batch_size, input_dtype, train=False, config=config
    )
    num_classes = dataset_builder.info.features["label"].num_classes

    if config.steps_per_eval == -1:
        num_validation_examples = dataset_builder.info.splits["validation"].num_examples
        steps_per_eval = math.ceil(num_validation_examples / config.batch_size)
    else:
        steps_per_eval = config.steps_per_eval

    logging.info("Step per eval : %d.", steps_per_eval)

    model_cls = getattr(models, config.model)
    model = create_model(
        model_cls=model_cls,
        half_precision=config.half_precision,
        num_classes=num_classes,
        init_stochastic_depth_rate=config.init_stochastic_depth_rate,
    )

    learning_rate_fn = create_learning_rate_fn(config, steps_per_epoch)
    state, with_batchnorm = create_eval_state(rngs, config, model, learning_rate_fn)

    checkpoint_manager = create_checkpoint_manager(workdir, config)
    state = restore_checkpoint(checkpoint_manager, state)
    state = jax_utils.replicate(state)

    logging.info("Restored checkpoint.")

    if not config.use_sync_batch_norm:
        logging.info("Sync batch satts.")
        state = sync_batch_stats(state)

    p_eval_step = jax.pmap(
        functools.partial(
            eval_step,
            num_classes=num_classes,
            with_batchnorm=with_batchnorm,
            model_ema=config.model_ema,
        ),
        axis_name="batch",
    )

    eval_metrics = []
    eval_metrics_last_t = time.time()

    for step in range(steps_per_eval):
        eval_batch = next(eval_iter)
        metrics = p_eval_step(state, eval_batch)
        eval_metrics.append(metrics)

        if config.get("log_every_steps"):
            if (step + 1) % config.log_every_steps == 0:
                inference_time = (
                    (time.time() - eval_metrics_last_t) / config.log_every_steps * 1000
                )

                logging.info(
                    "eval %d / %d, inference time = %.2f ms per %d batch",
                    (step + 1),
                    steps_per_eval,
                    inference_time,
                    local_batch_size,
                )
                eval_metrics_last_t = time.time()

    eval_metrics = common_utils.get_metrics(eval_metrics)
    summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
    logging.info(
        "Eval: loss: %.4f, top-1 accuracy: %.2f, top-5 accuracy: %.2f",
        summary["loss"],
        summary["top-1 accuracy"] * 100,
        summary["top-5 accuracy"] * 100,
    )

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state
