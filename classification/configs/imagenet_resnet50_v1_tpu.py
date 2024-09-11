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
    config.model = "ResNet50"

    # dataset
    config.dataset = "imagenet2012:5.*.*"

    # optimizer config
    config.optimizer = "sgd"
    config.learning_rate = 0.8
    config.warmup_epochs = 5.0
    config.momentum = 0.9

    config.cache = True
    config.half_precision = True
    config.batch_size = 2048
    config.shuffle_buffer_size = 16 * 1024

    config.num_epochs = 100

    config.init_stochastic_depth_rate = 0.0

    return config
