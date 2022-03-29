#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from typing import Any, Callable, Tuple

import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any


def _make_divisible(v, divisor=8, min_value=None):
    """Rifference:
    https://github.com/keras-team/keras/blob/v2.8.0/keras/applications/mobilenet_v2.py#L505
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResBlock(nn.Module):
    """Inverted ResNet block."""

    expansion: int
    strides: Tuple[int, int]
    alpha: float
    filters: int
    block_id: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        prefix = "block_{}_".format(self.block_id)

        inputs = x
        in_channels = x.shape[-1]
        pointwise_conv_filters = int(self.filters * self.alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

        # Expand
        x = self.conv(
            features=in_channels * self.expansion,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            name=prefix + "expand",
        )(x)
        x = self.norm(name=prefix + "expand_bn")(x)
        x = self.act(x)

        # Depthwise
        dw_filters = x.shape[-1]
        x = self.conv(
            features=dw_filters,
            kernel_size=(3, 3),
            strides=self.strides,
            padding="SAME",
            feature_group_count=dw_filters,
            name=prefix + "depthwise",
        )(x)
        x = self.norm(name=prefix + "depthwise_bn")(x)
        x = self.act(x)

        # Project
        x = self.conv(
            features=pointwise_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            name=prefix + "project",
        )(x)
        x = self.norm(name=prefix + "project_bn")(x)

        if in_channels == pointwise_filters and self.strides == 1:
            x = x + inputs

        return x
