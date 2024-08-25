#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp

from mobilenet_v2 import MobileNetV2

"""Tests for mobilenet_v2."""


jax.config.update("jax_disable_most_optimizations", True)


class MobileNetV2Test(parameterized.TestCase):
    """Test cases for MobileNet V2 model definition."""

    def test_mobilenet_v2_model(self):
        """Tests MobileNet V2 model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = MobileNetV2(alpha=1.0, num_classes=10, dtype=jnp.float32)
        variables = model_def.init(rng, jnp.ones((8, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)
        # MobileNet model will create parameters for the following layers:
        #   conv + batch_norm = 2
        #   InvertedResBlock layer = 17
        #   conv + batch_norm = 2
        #   Followed by a Dense layer = 1
        self.assertLen(variables["params"], 2)


if __name__ == "__main__":
    absltest.main()
