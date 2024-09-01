#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp

from mobilenet_v3 import MobileNetV3, Large, Small
from clu import parameter_overview

"""Tests for mobilenet_v3."""

jax.config.update("jax_disable_most_optimizations", True)


class MobileNetV3Test(parameterized.TestCase):
    """Test cases for MobileNet V3 model definition."""

    def test_mobilenet_v3_large_model(self):
        """Tests MobileNet V3 Large model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = MobileNetV3(
            alpha=1.0,
            num_classes=1000,
            layers=Large,
            last_block_filters=1280,
            dtype=jnp.float32,
        )
        variables = model_def.init(rng, jnp.ones((8, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)
        # MobileNet model will create parameters for the following layers:
        #   conv + batch_norm = 2
        #   InvertedResBlock layer = 15
        #   conv + batch_norm = 2
        #   conv = 2
        self.assertLen(variables["params"], 3)

        print(parameter_overview.get_parameter_overview(variables))

    def test_mobilenet_v3_small_model(self):
        """Tests MobileNet V3 Small model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = MobileNetV3(
            alpha=1.0,
            num_classes=1000,
            layers=Small,
            last_block_filters=1024,
            dtype=jnp.float32,
        )
        variables = model_def.init(rng, jnp.ones((8, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)
        # MobileNet model will create parameters for the following layers:
        #   conv + batch_norm = 2
        #   InvertedResBlock layer = 11
        #   conv + batch_norm = 2
        #   conv = 2
        self.assertLen(variables["params"], 3)

        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
