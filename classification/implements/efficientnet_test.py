#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
import jax
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from efficientnet import EfficientNet
import flax.linen as nn

"""Tests for EfficientNet."""


# EfficientNet B0
EfficientNet_B0 = partial(
    EfficientNet,
    width_coefficient=1.0,
    depth_coefficient=1.0,
    default_size=224,
    dropout_rate=0.2,
    init_stochastic_depth_rate=0.1,
)

# EfficientNet B1
EfficientNet_B1 = partial(
    EfficientNet,
    width_coefficient=1.0,
    depth_coefficient=1.1,
    default_size=240,
    dropout_rate=0.2,
    init_stochastic_depth_rate=0.1,
)

# jax.config.update("jax_disable_most_optimizations", True)


class EfficientNetTest(parameterized.TestCase):
    """Test cases for ConvNeXt model definition."""

    def test_efficientnet_b0_model(self):
        """Tests EfficientNet T model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = EfficientNet_B0(num_classes=1000, dtype=jnp.float32)
        variables = model_def.init(rng, jnp.ones((1, 224, 224, 3), jnp.float32))
        self.assertLen(variables, 2)

    def test_efficientnet_b1_model(self):
        """Tests ConvNeXt T model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = EfficientNet_B1(num_classes=1000, dtype=jnp.float32)
        variables = model_def.init(rng, jnp.ones((1, 224, 224, 3), jnp.float32))
        self.assertLen(variables, 2)

    def test_efficientnet_b0_summary(self):
        rng = jax.random.PRNGKey(0)
        model_def = EfficientNet_B0(num_classes=1000, dtype=jnp.float32)
        input_shape = (1, 224, 224, 3)

        tabulate_fn = nn.tabulate(
            model_def, rng, compute_flops=True, compute_vjp_flops=True
        )
        x = jnp.ones(input_shape, jnp.float32)
        print(tabulate_fn(x, train=False))


if __name__ == "__main__":
    absltest.main()
