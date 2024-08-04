#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
import math

import functools
import time
from typing import Dict

import input_pipeline
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow_datasets as tfds
from absl import logging

from flax import linen as nn
from flax import jax_utils
from flax.training import common_utils
from flax.training import train_state


""" Training Image classfication model.

    based on:
        https://github.com/google/flax/blob/main/examples/imagenet/models.py
"""


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


def initialized(key, model):
    input_shape = (1, 28, 28, 1)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init(key, jnp.ones(input_shape))
    params = variables["params"]

    return params


def cross_entropy_loss(logits, labels, num_classes):
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def compute_metrics(logits, labels, num_classes):
    loss = cross_entropy_loss(logits, labels, num_classes)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {"loss": loss, "accuracy": accuracy}
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


def train_step(
    state,
    batch,
    num_classes,
):
    """Perform a single training step."""

    def loss_fn(params):
        """loss function used for training."""
        logits = state.apply_fn(
            {"params": params},
            batch["image"],
        )

        loss = cross_entropy_loss(logits, batch["label"], num_classes)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name="batch")

    logits = aux[1]

    metrics = compute_metrics(logits, batch["label"], num_classes)
    new_state = state.apply_gradients(grads=grads)

    return new_state, metrics


def eval_step(state, batch, num_classes):
    variables = {"params": state.params}
    logits = state.apply_fn(variables, batch["image"], mutable=False)
    return compute_metrics(logits, batch["label"], num_classes)


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


def create_input_iter(dataset_builder, batch_size, train):
    ds = input_pipeline.create_split(
        dataset_builder,
        batch_size=batch_size,
        train=train,
    )
    it = map(prepare_tf_data, ds)
    it = jax_utils.prefetch_to_device(it, 2)
    return it


def create_train_state(
    rngs: Dict[str, jnp.ndarray],
    config: ml_collections.ConfigDict,
    model,
):
    """Create initial training state."""

    params = initialized(rngs, model)

    optimiser = optax.sgd(
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        nesterov=True,
    )

    logging.info(
        "Batch size: %d, Gradient accumulation steps : %d",
        config.batch_size,
        config.gradient_accumulation_steps,
    )
    tx = optax.MultiSteps(
        optimiser, every_k_schedule=config.gradient_accumulation_steps
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    return state


def train_and_evaluate(config: ml_collections.ConfigDict):
    """Execute model training and evaluation loop.
    Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the tensorboard summaries are written to.
    """

    rng = jax.random.PRNGKey(seed=config.seed)

    if config.batch_size % jax.device_count() > 0:
        raise ValueError("Batch size must be divisible by the number of devices")

    local_batch_size = config.batch_size // jax.process_count()

    dataset_builder = tfds.builder(
        config.dataset,
        data_dir=config.dataset_dir,
    )
    dataset_builder.download_and_prepare()
    train_iter = create_input_iter(dataset_builder, local_batch_size, train=True)
    eval_iter = create_input_iter(dataset_builder, local_batch_size, train=False)
    num_classes = dataset_builder.info.features["label"].num_classes

    steps_per_epoch = math.ceil(
        dataset_builder.info.splits["train"].num_examples
        / (config.batch_size * config.gradient_accumulation_steps)
    )
    steps_per_epoch *= config.gradient_accumulation_steps
    steps_per_epoch = (
        steps_per_epoch if steps_per_epoch % 2 == 0 else steps_per_epoch + 1
    )

    if config.num_train_steps <= 0:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps

    if config.steps_per_eval == -1:
        num_test_examples = dataset_builder.info.splits["test"].num_examples
        steps_per_eval = math.ceil(num_test_examples / config.batch_size)

    else:
        steps_per_eval = config.steps_per_eval

    logging.info(
        "Steps per epech : %d, Step per eval : %d, Num steps : %d",
        steps_per_epoch,
        steps_per_eval,
        num_steps,
    )

    model = CNN()

    state = create_train_state(rng, config, model)

    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            num_classes=num_classes,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        ),
        axis_name="batch",
    )
    p_eval_step = jax.pmap(
        functools.partial(eval_step, num_classes=num_classes),
        axis_name="batch",
    )

    train_metrics_last_t = time.time()
    logging.info("Initial compilation, this might take some minutes...")

    for step, batch in zip(range(step_offset, num_steps), train_iter):
        state, metrics = p_train_step(state, batch)

        # logging.info(state)

        if step == step_offset:
            logging.info("Initial compilation completed.")

        if config.get("log_every_steps"):
            train_metrics = []
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
                logging.info(
                    "train steps: %d, loss: %.4f, accuracy: %.2f",
                    (step + 1),
                    summary["train_loss"],
                    summary["train_accuracy"] * 100,
                )

        if (step + 1) % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            eval_metrics = []

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

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state
