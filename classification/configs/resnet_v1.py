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
    config.model = "ResNet50"

    config.optimizer = "adamw"
    config.learning_rate = 0.004
    config.adamw_weight_decay = 0.05
    config.l2_weight_decay = 0.0
    config.warmup_epochs = 20.0
    config.momentum = 0.9
    config.batch_size = 2048
    config.label_smoothing = 0.1
    config.model_ema_decay = 0.9999
    config.model_ema = True

    config.num_epochs = 300

    # randomarug
    config.aug_type = "randaug"
    config.randaug_num_layers = 2
    config.randaug_magnitude = 9
    config.randaug_cutout_const = 40.0
    config.randaug_translate_const = 250
    config.randaug_magnitude_std = 0.5
    config.randaug_prob_to_apply = None
    config.randaug_exclude_ops = None

    # random erasing
    config.random_erasing = False

    # mixup and cutmix
    config.mixup_and_cutmix = True
    config.mixup_and_cutmix_mixup_alpha = 0.8
    config.mixup_and_cutmix_cutmix_alpha = 1.0
    config.mixup_and_cutmix_prob = 1.0
    config.mixup_and_cutmix_switch_prob = 0.5
    config.mixup_and_cutmix_label_smoothing = 0.1

    config.init_stochastic_depth_rate = 0.1

    return config
