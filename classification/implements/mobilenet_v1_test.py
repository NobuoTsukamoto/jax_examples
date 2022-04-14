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

from mobilenet_v1 import MobileNetV1
from clu import parameter_overview

"""Tests for mobilenet_v1."""


jax.config.update("jax_disable_most_optimizations", True)


class MobileNetV1Test(parameterized.TestCase):
    """Test cases for MobileNet V1 model definition."""

    def test_mobilenet_v1_model(self):
        """Tests MobileNet V1 model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = MobileNetV1(
            alpha=1.0, depth_multiplier=1.0, num_classes=1000, dtype=jnp.float32
        )
        variables = model_def.init(rng, jnp.ones((8, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)
        # MobileNet model will create parameters for the following layers:
        #   conv + batch_norm = 2
        #   DepthwiseSeparable layer = 13
        #   conv = 1
        self.assertLen(variables["params"], 16)

        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
