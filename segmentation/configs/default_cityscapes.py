#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
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
    config.half_precision = False

    # fmt: off
    config.class_weights = [
        0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
        1.0865, 1.0955, 1.0865, 1.1529, 1.0507,
    ]
    # config.class_weights = None
    # fmt: on
    config.weight_decay = None

    # Data augmentation
    config.image_size = (1024, 2048)
    config.min_resize_value = 0.5
    config.max_resize_value = 2.0
    config.crop_image_size = (512, 1024)

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    config.ignore_label = 255

    config.loss = "cross_entropy_loss"
    config.label_smoothing = 0.0

    return config
