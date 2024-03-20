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
from common_layer import BottleneckResNetBlock, BottleneckConvNeXtBlock

"""Tests for convnext."""


jax.config.update("jax_disable_most_optimizations", True)

ConvNeXt_T = partial(
    ConvNeXt,
    stage_sizes=[3, 3, 9, 3],
    num_filters=[64, 128, 256, 512],
    block_cls=BottleneckConvNeXtBlock,
)

ConvNeXt_T_patchify_stem = partial(
    ConvNeXt,
    stage_sizes=[3, 3, 9, 3],
    num_filters=[64, 128, 256, 512],
    block_cls=BottleneckResNetBlock,
)

ConvNeXt_T_ResNeXtify = partial(
    ConvNeXt,
    stage_sizes=[3, 3, 9, 3],
    num_filters=[96, 192, 384, 768],
    block_cls=BottleneckResNetBlock,
)


class ConvNeXtTest(parameterized.TestCase):
    """Test cases for ConvNeXt model definition."""

    def test_convnext_t_model(self):
        """Tests ConvNeXt T model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = ConvNeXt_T(num_classes=10, dtype=jnp.float32)
        variables = model_def.init(rng, jnp.ones((1, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)

        # Total: 26,654,474
        print(parameter_overview.get_parameter_overview(variables))

    def test_convnext_t_patchify_stem_model(self):
        """Tests ConvNeXt T model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = ConvNeXt_T_patchify_stem(num_classes=10, dtype=jnp.float32)
        variables = model_def.init(rng, jnp.ones((1, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)

        # Total: 13,754,954
        print(parameter_overview.get_parameter_overview(variables))

    def test_convnext_t_resnextify_model(self):
        """Tests ConvNeXt T model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = ConvNeXt_T_ResNeXtify(num_classes=10, dtype=jnp.float32)
        variables = model_def.init(rng, jnp.ones((1, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)

        # Total: 59,846,666
        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
