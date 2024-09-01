#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import ml_collections
import optax
from absl import logging


def create_learning_rate_fn(config: ml_collections.ConfigDict, steps_per_epoch: int):
    """Create learning rate schedule."""

    warmup_steps = config.warmup_epochs * (
        steps_per_epoch // config.gradient_accumulation_steps
    )
    decay_steps = config.num_epochs * (
        steps_per_epoch // config.gradient_accumulation_steps
    )

    logging.info(
        "Learning Rate Scheduler: %s, init_value=%f, peak_value=%f, warmup_steps=%d, "
        "decay_steps=%d, transition_steps=%d, decay_rate=%f, transition_begin=%d, "
        "staircase=%s",
        config.optimizer_schedule,
        config.initial_learning_rate,
        config.learning_rate,
        warmup_steps,
        decay_steps,
        config.transition_steps,
        config.exponential_decay_rate,
        warmup_steps,
        config.lr_drop_staircase,
    )

    if config.optimizer_schedule == "warmup_exponential_decay":
        schedule_fn = optax.warmup_exponential_decay_schedule(
            init_value=config.initial_learning_rate,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            transition_steps=config.transition_steps,
            decay_rate=config.exponential_decay_rate,
            transition_begin=warmup_steps,
            staircase=config.lr_drop_staircase,
        )

    elif config.optimizer_schedule == "warmup_cosine_decay":
        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=config.initial_learning_rate,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=config.end_learning_rate,
        )

    elif config.optimizer_schedule == "cosine":
        schedule_fn = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=decay_steps,
        )

    else:
        schedule_fn = optax.constant_schedule(config.learning_rate)

    return schedule_fn


def create_optimizer(config: ml_collections.ConfigDict, learning_rate_fn):
    """Create optimizer."""

    logging.info(
        "Optimizer: %s, weight_decay=%f",
        config.optimizer,
        config.weight_decay,
    )

    if config.optimizer == "adamw":
        tx = optax.adamw(
            learning_rate=learning_rate_fn, weight_decay=config.weight_decay
        )

    elif config.optimizer == "rmsprop":
        tx = optax.rmsprop(
            learning_rate=learning_rate_fn,
            decay=config.rmsprop_decay,
            momentum=config.momentum,
            eps=config.rmsprop_epsilon,
        )

    elif config.optimizer == "sgd":
        tx = optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=config.momentum,
            nesterov=True,
        )

    if config.model_ema_decay > 0.0:
        logging.info(
            "Decay rate for the exponential moving average. : %f",
            config.model_ema_decay,
        )

    if config.gradient_accumulation_steps > 1:
        logging.info(
            "Gradient accumulation steps. : %d",
            config.gradient_accumulation_steps,
        )
        tx = optax.MultiSteps(tx, config.gradient_accumulation_steps)

    return tx