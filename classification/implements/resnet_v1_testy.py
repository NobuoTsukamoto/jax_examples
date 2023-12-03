#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
import jax
from absl.testing import absltest, parameterized
from clu import parameter_overview
from jax import numpy as jnp
from resnet_v1 import ResNet
from common_layer import ResNetBlock, BottleneckResNetBlock

"""Tests for resnet_v1."""


jax.config.update("jax_disable_most_optimizations", True)

ResNet18 = partial(
    ResNet,
    stage_sizes=[2, 2, 2, 2],
    num_filters=[64, 128, 256, 512],
    block_cls=ResNetBlock,
)
ResNet50 = partial(
    ResNet,
    stage_sizes=[3, 4, 6, 3],
    num_filters=[64, 128, 256, 512],
    block_cls=BottleneckResNetBlock,
)


class ResNetTest(parameterized.TestCase):
    """Test cases for ResNet model definition."""

    def test_resnet_v1_18_model(self):
        """Tests ResNet18 model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = ResNet18(num_classes=10, dtype=jnp.float32)
        variables = model_def.init(rng, jnp.ones((1, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)
        # ResNet 18 model will create parameters for the following layers:
        #   layer1 : 1
        #   layer2 : 3
        #   layer3 : 4
        #   layer4 : 6
        #   layer4 : 3
        #   Followed by a Dense layer = 2
        # self.assertLen(variables["params"], 19)

        # Total: 11,187,138
        print(parameter_overview.get_parameter_overview(variables))

    def test_resnet_v1_50_model(self):
        """Tests ResNet50 model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = ResNet50(num_classes=10, dtype=jnp.float32)
        variables = model_def.init(rng, jnp.ones((1, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)
        # ResNet 18 model will create parameters for the following layers:
        #   layer1 : 1
        #   layer2 : 3
        #   layer3 : 4
        #   layer4 : 6
        #   layer4 : 3
        #   Followed by a Dense layer = 2
        # self.assertLen(variables["params"], 19)

        # Total: 23,581,642
        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
