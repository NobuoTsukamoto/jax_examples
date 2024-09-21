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

    # `name` argument of tensorflow_datasets.builder()
    config.dataset = "cityscapes:1.*.*"
    config.dataset_dir = "/workdir/tensorflow_datasets/"
    config.num_classes = 19
    config.model_input = (512, 1024)

    config.cache = False
    config.shuffle_buffer_size = 8 * 128
    config.prefetch = 10
    config.half_precision = False

    # fmt: off
    config.class_weights = [
        0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
        1.0865, 1.0955, 1.0865, 1.1529, 1.0507,
    ]
    # config.class_weights = None
    # fmt: on
    config.weight_decay = 0.00004

    # Data augmentation
    config.image_size = (1024, 2048)
    config.min_resize_value = 0.5
    config.max_resize_value = 2.0
    config.crop_image_size = (512, 1024)
    config.output_image_size = (512, 1024)

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    config.ignore_label = 255

    config.loss = "cross_entropy_loss"
    config.label_smoothing = 0.0

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

    config.model_ema = False
    config.model_ema_decay = 0.0

    config.gradient_accumulation_steps = 1

    return config
