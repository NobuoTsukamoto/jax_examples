#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from absl.testing import absltest
from absl.testing import parameterized

import tensorflow as tf

# Local imports.
import summary
from configs import default as default_lib


class TrainTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        # Make sure tf does not allocate gpu memory.
        tf.config.experimental.set_visible_devices([], "GPU")

    @parameterized.product(model=("MobileNetV2_10",))
    def test_summrize(self, model):
        """Tests training and evaluation loop using mocked data."""
        # Create a temporary directory where tensorboard metrics are written.

        # Define training configuration
        config = default_lib.get_config()
        config.model = model

        summary.summarize(config=config)


if __name__ == "__main__":
    absltest.main()
