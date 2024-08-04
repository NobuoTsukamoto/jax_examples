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
    config.dataset = "mnist"
    config.dataset_dir = "/workdir/tensorflow_datasets/"

    config.learning_rate = 0.1
    config.momentum = 0.9
    config.batch_size = 4

    config.num_epochs = 1
    config.log_every_steps = 100

    config.gradient_accumulation_steps = 2

    config.seed = 42

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1
    return config
