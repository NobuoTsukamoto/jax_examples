#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from configs import default_cityscapes as default_lib


def get_config():
    """Get the default hyperparameter configuration."""
    config = default_lib.get_config()

    # As defined in the `models` module.
    config.model = "Fast_SCNN"

    # optimizer
    config.optimizer = "sgd"
    config.momentum = 0.9
    config.l2_weight_decay = 0.00004

    # LR scheduler config
    config.optimizer_schedule = "warmup_cosine_decay"
    config.initial_learning_rate = 0.0
    config.warmup_epochs = 10.0
    config.learning_rate = 0.01
    config.end_learning_rate = 0.0

    config.batch_size = 8
    config.num_epochs = 1000

    return config
