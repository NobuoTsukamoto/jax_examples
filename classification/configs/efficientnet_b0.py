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
    config.model = "EfficientNet_B0"

    # preprocessing
    config.input_pipeline_type = "efficientnet"

    # optimizer config
    config.optimizer = "rmsprop"
    config.rmsprop_decay = 0.9
    config.rmsprop_epsilon = 0.001
    config.momentum = 0.9
    config.rmsprop_initial_scale = 1.0

    # LR scheduler config
    config.optimizer_schedule = "warmup_exponential_decay"
    config.initial_learning_rate = 0.0
    config.learning_rate = 0.001  # 0.016 * batch_size(16) / 256
    config.warmup_epochs = 5
    config.exponential_decay_rate = 0.97
    config.transition_steps = 1291  # 2.4 * steps_per_epoch (538)
    config.lr_drop_staircase = True

    # Auto augment
    config.aug_type = "autoaug"
    config.autoaug_augmentation_name = "v0"
    config.autoaug_cutout_const = 100
    config.autoaug_translate_const = 250

    config.cache = False
    config.half_precision = False
    config.batch_size = 16

    config.label_smoothing = 0.1
    config.l2_weight_decay = 1e-5
    config.weight_decay_exclude_layers = [
        "BatchNorm",
    ]

    # model ema
    config.model_ema = True
    config.model_ema_decay = 0.9999
    config.model_ema_type = "v2"
    config.model_ema_debias = False
    config.model_ema_trainable_weights_only = False

    config.use_sync_batch_norm = False

    config.num_epochs = 350

    return config
