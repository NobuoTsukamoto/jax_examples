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

    # As defined in the `models` module.
    config.model = "FCN_ResNetV2"
    config.image_size = (1024, 2048)

    # `name` argument of tensorflow_datasets.builder()
    config.dataset = "cityscapes:1.*.*"
    config.num_classes = 19

    # Training param
    config.optimizer = "adam"
    config.learning_rate = 0.1
    config.batch_size = 4

    config.num_epochs = 100.0
    config.log_every_steps = 200

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

    # Data augmentation
    config.min_resize_value = 0.25
    config.max_resize_value = 0.75
    config.output_image_size = (512, 1024)

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    config.ignore_label = 255
    return config
