#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from typing import Any

import jax.numpy as jnp
from flax import linen as nn


class LayerScale(nn.Module):
    """Create a layer scale.

        Reference
            - [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239v2)

    Attributes:
        init_values:
        projection_dim:
    """

    projection_dim: int
    init_values: float = 1e-6
    dtype: Any = jnp.float32

    def setup(self):
        self.scale = (
            self.param(
                "scale",
                nn.initializers.ones,
                (self.projection_dim,),
                self.dtype,
            )
            * self.init_values
        )

    @nn.compact
    def __call__(self, inputs):
        return inputs * self.scale
