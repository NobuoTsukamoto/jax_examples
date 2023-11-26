#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from configs import default as default_lib


def get_config():
    """Get the default hyperparameter configuration."""
    config = default_lib.get_config()

    # As defined in the `models` module.
    config.model = "MobileNetV3_Large"

    config.learning_rate = 0.1
    config.warmup_epochs = 5.0
    config.momentum = 0.9
    config.batch_size = 64

    # config.mixup_and_cutmix = True

    config.num_epochs = 100.0

    return config
