#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # As defined in the `models` module.
    config.model = "ResNet50"
    config.image_size = 224

    # `name` argument of tensorflow_datasets.builder()
    config.dataset = "imagenette/full-size-v2:1.*.*"
    config.dataset_dir = "/workdir/tensorflow_datasets/"

    # optimizer
    config.optimizer = "sgd"
    config.optimizer_schedule = "warmup_cosine_decay"
    config.weight_decay = 0.0001
    config.initial_learning_rate = 0.0
    config.learning_rate = 0.1
    config.end_learning_rate = 0.0
    config.exponential_decay_rate = 0.0
    config.warmup_epochs = 5.0
    config.momentum = 0.9
    config.batch_size = 64
    config.label_smoothing = 0.0
    config.l2_weight_decay = 0.0001
    config.transition_steps = 0
    config.lr_drop_staircase = False
    config.adam_epsilon = 1e-8

    config.num_epochs = 100.0
    config.log_every_steps = 100

    config.cache = False
    config.shuffle_buffer_size = 16 * 4096
    config.prefetch = 10

    config.half_precision = False

    config.model_ema = False
    config.model_ema_type = "v1" # v1, v2
    config.model_ema_decay = 0.0
    config.model_ema_trainable_weights_only = True

    config.gradient_accumulation_steps = 1

    # Input and augmentation
    config.aug_rand_horizontal_flip = True
    config.aug_type = "none"  # randaug, autoaug, none

    # autoaug
    config.autoaug_augmentation_name = "v0"
    config.autoaug_cutout_const = 100.0
    config.autoaug_translate_const = 250.0

    # randomarug
    config.randaug_num_layers = 2
    config.randaug_magnitude = 9
    config.randaug_cutout_const = 40.0
    config.randaug_translate_const = 0.0
    config.randaug_magnitude_std = 0.0
    config.randaug_prob_to_apply = None
    config.randaug_exclude_ops = None

    config.color_jitter = 0.0

    # random erasing
    config.random_erasing = False
    config.random_erasing_probability = 0.25
    config.random_erasing_min_area = 0.02
    config.random_erasing_max_area = 1 / 3
    config.random_erasing_min_aspect = 0.3
    config.random_erasing_max_aspect = None
    config.random_erasing_min_count = 1
    config.random_erasing_max_count = 1
    config.random_erasing_trials = 10

    config.mixup_and_cutmix = False
    config.mixup_and_cutmix_mixup_alpha = 0.8
    config.mixup_and_cutmix_cutmix_alpha = 1.0
    config.mixup_and_cutmix_prob = 1.0
    config.mixup_and_cutmix_switch_prob = 0.5
    config.mixup_and_cutmix_label_smoothing = 0.1

    config.crop_area_range = (0.08, 1.0)
    config.center_crop_fraction = 0.875

    config.three_augment = False

    config.init_stochastic_depth_rate = 0.0

    config.seed = 42

    config.max_to_keep_checkpoint = 5

    config.profile = True

    config.use_sync_batch_norm = True

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1
    return config
