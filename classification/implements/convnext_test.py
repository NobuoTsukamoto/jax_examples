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
from clu import parameter_overview
from jax import numpy as jnp
from convnext import ConvNeXt
import flax.linen as nn
from common_layer import BottleneckResNetBlock, BottleneckConvNeXtBlock

"""Tests for convnext."""


# jax.config.update("jax_disable_most_optimizations", True)

ConvNeXt_T = partial(
    ConvNeXt,
    stage_sizes=[3, 3, 9, 3],
    num_filters=[96, 192, 384, 768],
    kernel_size=(7, 7),
    block_cls=BottleneckConvNeXtBlock,
)


class ConvNeXtTest(parameterized.TestCase):
    """Test cases for ConvNeXt model definition."""

    def test_convnext_t_model(self):
        """Tests ConvNeXt T model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = ConvNeXt_T(num_classes=1000, dtype=jnp.float32)
        tabulate_fn = nn.tabulate(
            model_def, rng, compute_flops=True, compute_vjp_flops=True
        )
        x = jnp.ones((1, 224, 224, 3), jnp.float32)
        
        print(tabulate_fn(x, train=False))

    def test_convnext_t_model2(self):
        """Tests ConvNeXt T model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = ConvNeXt_T(num_classes=1000, dtype=jnp.float32)
        variables = model_def.init(rng, jnp.ones((1, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 1)

        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
