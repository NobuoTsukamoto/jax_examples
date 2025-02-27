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
    config.model = "MobileNetV3_Large"

    # dataset
    config.dataset = "imagenet2012:5.*.*"

    # optimizer config
    config.optimizer = "adamw"
    config.adam_epsilon = 1e-7
    config.weight_decay = 0.1
    config.l2_weight_decay = 0.0

    # Auto augment
    config.aug_type = "autoaug"
    config.autoaug_augmentation_name = "v0"
    config.autoaug_cutout_const = 100
    config.autoaug_translate_const = 250

    # LR scheduler config
    config.optimizer_schedule = "cosine"
    config.learning_rate = 0.004

    config.cache = True
    config.half_precision = True

    config.batch_size = 4096

    config.model_ema = True
    config.model_ema_decay = 0.9999
    config.model_ema_type = "v2"
    config.model_ema_trainable_weights_only = False

    config.num_epochs = 700

    config.use_sync_batch_norm = False

    return config
