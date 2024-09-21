#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import jax
import jax.numpy as jnp
import ml_collections
import models
from jax import random
import flax.linen as nn


""" Sammarize semantic segmentation model.
"""


def create_model(*, model_cls, half_precision, num_classes, output_size, **kwargs):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == "tpu":
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_cls(
        num_classes=num_classes, output_size=output_size, dtype=model_dtype, **kwargs
    )


def summarize(config: ml_collections.ConfigDict):
    """Summarize model.
    Args:
        config: Hyperparameter configuration for training and evaluation.
    """

    rng = random.PRNGKey(0)

    num_classes = config.num_classes
    output_size = config.output_image_size

    model_cls = getattr(models, config.model)
    model = create_model(
        model_cls=model_cls,
        half_precision=config.half_precision,
        num_classes=num_classes,
        output_size=output_size,
    )
    input_shape = (1, config.model_input[0], config.model_input[1], 3)

    tabulate_fn = nn.tabulate(model, rng, compute_flops=True, compute_vjp_flops=True)
    x = jnp.ones(input_shape, jnp.float32)

    print(tabulate_fn(x, train=False))
