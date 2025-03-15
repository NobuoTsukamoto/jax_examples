#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2025 Nobuo Tsukamoto
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

    # randomarug
    config.aug_type = "randaug"
    config.randaug_num_layers = 2
    config.randaug_magnitude = 15
    config.randaug_cutout_const = 20
    config.randaug_translate_const = 10
    config.randaug_magnitude_std = 0.0
    config.randaug_prob_to_apply = 0.7
    config.randaug_exclude_ops = ["Cutout"]

    # optimizer
    config.optimizer = "adamw"
    config.weight_decay = 0.01
    config.adam_epsilon = 1e-7

    # LR scheduler config
    config.optimizer_schedule = "warmup_cosine_decay"
    config.learning_rate = 0.004
    config.warmup_epochs = 5

    config.cache = True
    config.half_precision = True

    config.batch_size = 4096

    config.num_epochs = 500

    config.use_sync_batch_norm = False

    return config
