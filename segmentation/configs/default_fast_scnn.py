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

    config.model_input = (512, 1024)
    config.output_image_size = (512, 1024)

    # optimizer
    config.optimizer = "sgd"
    config.momentum = 0.9

    # LR scheduler config
    config.optimizer_schedule = "warmup_cosine_decay"
    config.initial_learning_rate = 0.0
    config.warmup_epochs = 10.0
    config.learning_rate = 0.045
    config.end_learning_rate = 0.0

    config.batch_size = 8

    config.weight_decay = 0.00004

    config.num_epochs = 1000
    config.log_every_steps = 100

    # Data augmentation
    config.min_resize_value = 0.5
    config.max_resize_value = 2.0

    return config
