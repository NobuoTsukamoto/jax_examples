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
    config.model = "MobileNetV3_Large"

    # dataset
    config.dataset = "imagenet2012:5.*.*"

    # optimizer config
    config.optimizer = "rmsprop"
    config.rmsprop_decay = 0.9
    config.rmsprop_epsilon = 0.002
    config.momentum = 0.9
    config.rmsprop_initial_scale = 1.0

    # LR scheduler config
    config.optimizer_schedule = "warmup_exponential_decay"
    config.initial_learning_rate = 0.0
    config.learning_rate = 0.426  # 0.02 * (batch_size / 192)
    config.warmup_epochs = 5
    config.exponential_decay_rate = 0.99
    config.transition_steps = 939  # 3.0 * steps_per_epoch (313)
    config.lr_drop_staircase = True

    config.cache = True
    config.half_precision = True
    config.batch_size = 4096  # 512 * 8

    config.label_smoothing = 0.1
    config.l2_weight_decay = 0.00001

    config.model_ema = True
    config.model_ema_decay = 0.9999
    config.model_ema_type = "v2"
    config.model_ema_trainable_weights_only = False

    config.num_epochs = 1000

    config.use_sync_batch_norm = True

    return config
