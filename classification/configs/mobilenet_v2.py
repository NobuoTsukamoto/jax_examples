#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from configs import default as default_lib


def get_config():
    """Get the default hyperparameter configuration."""
    config = default_lib.get_config()

    # As defined in the `models` module.
    config.model = "MobileNetV2_10"

    # optimizer config
    config.optimizer = "rmsprop"
    config.rmsprop_decay = 0.9
    config.rmsprop_epsilon = 0.002
    config.momentum = 0.9
    config.rmsprop_initial_scale = 1.0

    # LR scheduler config
    config.optimizer_schedule = "warmup_exponential_decay"
    config.initial_learning_rate = 0.0
    config.learning_rate = 0.001  # 0.008 * batch_size(16) / 128
    config.warmup_epochs = 5
    config.exponential_decay_rate = 0.94
    config.transition_steps = 1345  # 2.5 * steps_per_epoch(538)
    config.lr_drop_staircase = True

    config.half_precision = False
    config.batch_size = 16

    config.label_smoothing = 0.1
    config.l2_weight_decay = 0.00001

    config.num_epochs = 500

    return config
