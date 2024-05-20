#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import jax
from absl.testing import absltest, parameterized
from clu import parameter_overview
from jax import numpy as jnp
from vit import VitInputLayer, MultiHeadSelfAttention

"""Tests for ViT."""


class ViTTest(parameterized.TestCase):
    """Test cases for ViT model definition."""

    def test_input_layer(self):
        """Tests ViT Input Layer definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = VitInputLayer(num_patch_row=2, dtype=jnp.float32)
        variables = model_def.init(rng, jnp.ones((2, 32, 32, 2), jnp.float32))

        self.assertLen(variables, 1)

        print(parameter_overview.get_parameter_overview(variables))

    def test_mhsa(self):
        """Tests ViT Multi Head Self Attention definition and output (variables)."""
        rngs = {
            "params": jax.random.PRNGKey(0),
            "dropout": jax.random.PRNGKey(1),
        }
        model_def = MultiHeadSelfAttention(dtype=jnp.float32)
        variables = model_def.init(rngs, jnp.ones((2, 5, 384), jnp.float32))

        self.assertLen(variables, 1)

        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()