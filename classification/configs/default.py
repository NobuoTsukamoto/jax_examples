#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # As defined in the `models` module.
    config.model = "MobileNetV3_Large"
    config.image_size = (224, 224)

    # `name` argument of tensorflow_datasets.builder()
    config.dataset = "imagenette/full-size-v2:1.*.*"

    config.learning_rate = 0.1
    config.warmup_epochs = 5.0
    config.momentum = 0.9
    config.batch_size = 64

    config.num_epochs = 100.0
    config.log_every_steps = 100

    config.cache = False
    config.half_precision = False

    # Input and augmentation
    config.aug_rand_horizontal_flip = True
    config.aug_type = "randaug"     # randaug, autoaug, None

    # autoaug
    config.autoaug_augmentation_name = "v0"
    config.autoaug_cutout_const = 100.
    config.autoaug_translate_const = 250.

    # randomarug
    config.randaug_num_layers = 2
    config.randaug_magnitude = 10.
    config.randaug_cutout_const = 40.
    config.randaug_translate_const = 0.
    config.randaug_prob_to_apply = None
    config.randaug_exclude_ops = ["Cutout"]

    config.color_jitter = 0.

    # random erasing
    config.random_erasing = True
    config.random_erasing.probability = 0.25
    config.random_erasing_min_area = 0.02
    config.random_erasing_max_area = 1 / 3
    config.random_erasing_min_aspect = 0.3
    config.random_erasing_max_aspect = None
    config.random_erasing_min_count = 1
    config.random_erasing_max_count = 1
    config.random_erasing_trials = 10

    config.crop_area_range = (0.08, 1.0)
    config.center_crop_fraction = 0.875

    config.three_augment = False

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1
    return config
