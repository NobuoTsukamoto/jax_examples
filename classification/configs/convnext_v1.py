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
    config.model = "ConvNeXt_T"

    config.optimizer = "adamw"
    config.learning_rate = 0.004
    config.adamw_weight_decay = 0.05
    config.warmup_epochs = 0.066
    config.momentum = 0.9
    config.batch_size = 64

    config.num_epochs = 300
    config.aug_type = "randaug"
    config.random_erasing = True
    config.mixup_and_cutmix = True

    config.init_stochastic_depth_rate = 0.2

    return config
