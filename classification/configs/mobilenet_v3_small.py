#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from configs import default as default_lib


def get_config():
    """Get the default hyperparameter configuration."""
    config = default_lib.get_config()

    # As defined in the `models` module.
    config.model = "MobileNetV3_Small"

    config.optimizer = "rmsprop"
    config.rmsprop_initial_scale = 1.0
    config.rmsprop_momentum = 0.9
    config.rmsprop_epsilon = 0.002
    config.rmsprop_decay = 0.9

    config.batch_size = 32
    config.label_smoothing = 0.1
    config.l2_weight_decay = 0.00001

    config.optimizer_schedule = "warmup_exponential_decay"
    config.initial_learning_rate = 0.0
    config.learning_rate = 0.003  # 0.02 * (32 / 192)
    config.warmup_epochs = 5
    config.exponential_decay_rate = 0.99
    config.transition_steps = 888  # 3.0 * steps_per_epoch(296)
    config.lr_drop_staircase = True

    config.num_epochs = 1000

    config.model_ema_decay = 0.9999
    config.model_ema = True

    return config
