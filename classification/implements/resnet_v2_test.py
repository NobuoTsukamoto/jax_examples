#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import jax
from absl.testing import absltest, parameterized
from clu import parameter_overview
from jax import numpy as jnp
from resnet_v2 import ResNet50V2_layer, ResNetV2

"""Tests for resnet_v2."""


jax.config.update("jax_disable_most_optimizations", True)


class ResNetV2Test(parameterized.TestCase):
    """Test cases for MobileNet V2 model definition."""

    def test_resnet_v2_model(self):
        """Tests MobileNet V2 model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = ResNetV2(
            layers=ResNet50V2_layer, num_classes=1000, dtype=jnp.float32
        )
        variables = model_def.init(rng, jnp.ones((1, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)
        # ResNet50 v2 model will create parameters for the following layers:
        #   layer1 : 1
        #   layer2 : 3
        #   layer3 : 4
        #   layer4 : 6
        #   layer4 : 3
        #   Followed by a Dense layer = 2
        self.assertLen(variables["params"], 19)

        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
