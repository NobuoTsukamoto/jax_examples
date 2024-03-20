#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from typing import Any, Callable, Tuple, Optional

import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, random
from flax import linen as nn

ModuleDef = Any


def _make_divisible(v, divisor=8, min_value=None):
    """
    Reference:
        https://github.com/keras-team/keras/blob/v2.8.0/keras/applications/mobilenet_v2.py#L505
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ResNetBlock(nn.Module):
    """ResNet block."""

    features: int
    conv: ModuleDef
    norm: ModuleDef
    stochastic_depth: ModuleDef
    act: Callable
    strides: Optional[Tuple[int, int]] = (1, 1)
    stochastic_depth_drop_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.features, kernel_size=(3, 3), strides=self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.features, kernel_size=(3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(self.features, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        if self.stochastic_depth_drop_rate > 0.0:
            y = self.stochastic_depth(
                stochastic_depth_drop_rate=self.stochastic_depth_drop_rate
            )(y)

        return self.act(residual + y)


class ResidualBlockV2(nn.Module):
    """Residual block v2"""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    name: str
    strides: Tuple[int, int] = None
    is_conv_shortcut: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        pre = self.norm(name=self.name + "_preact_bn")(inputs)
        pre = self.act(pre)

        if self.is_conv_shortcut:
            shortcut = self.conv(
                features=self.filters * 4,
                kernel_size=(1, 1),
                strides=self.strides,
                padding="SAME",
                name=self.name + "_0_conv",
            )(pre)

        elif self.strides is not None:
            shortcut = nn.max_pool(pre, (1, 1), strides=self.strides, padding="SAME")

        else:
            shortcut = pre

        x = self.conv(
            features=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            name=self.name + "_1_conv",
        )(pre)
        x = self.norm(name=self.name + "_1_bn")(x)
        x = self.act(x)

        x = self.conv(
            features=self.filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name=self.name + "_2_conv",
        )(x)
        x = self.norm(name=self.name + "_2_bn")(x)
        x = self.act(x)

        x = self.conv(
            features=self.filters * 4,
            kernel_size=(1, 1),
            strides=self.strides,
            padding="SAME",
            name=self.name + "_3_conv",
        )(x)
        x = shortcut + x

        return x


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    features: int
    conv: ModuleDef
    norm: ModuleDef
    stochastic_depth: ModuleDef
    act: Callable
    strides: Optional[Tuple[int, int]] = (1, 1)
    stochastic_depth_drop_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.features, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.features, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.features * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.features * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        if self.stochastic_depth_drop_rate > 0.0:
            y = self.stochastic_depth(
                stochastic_depth_drop_rate=self.stochastic_depth_drop_rate
            )(y)
        return self.act(residual + y)


class BottleneckConvNeXtBlock(nn.Module):
    """Bottleneck ConvNeXt block."""

    features: int
    conv: ModuleDef
    norm: ModuleDef
    stochastic_depth: ModuleDef
    act: ModuleDef
    strides: Optional[Tuple[int, int]] = (1, 1)
    stochastic_depth_drop_rate: Optional[float] = 0.0
    kernel_size: Optional[Tuple[int, int]] = (7, 7)

    @nn.compact
    def __call__(self, x):
        residual = x
        # Depthwise
        dw_filters = x.shape[-1]
        y = self.conv(
            features=dw_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="SAME",
            feature_group_count=dw_filters,
        )(x)
        y = self.norm()(y)

        y = self.conv(self.features * 4, (1, 1))(y)
        y = self.norm()(y)
        y = self.act(y)

        y = self.conv(self.features, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(self.features, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        if self.stochastic_depth_drop_rate > 0.0:
            y = self.stochastic_depth(
                stochastic_depth_drop_rate=self.stochastic_depth_drop_rate
            )(y)
        return residual + y


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
    def __call__(self, inputs):
        prefix = "block_{}_".format(self.block_id)

        in_channels = inputs.shape[-1]
        pointwise_conv_filters = int(self.filters * self.alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

        # Expand
        x = self.conv(
            features=in_channels * self.expansion,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            name=prefix + "expand",
        )(inputs)
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

        if in_channels == pointwise_filters and self.strides == (1, 1):
            x = x + inputs

        return x


class InvertedResBlockMobileNetV3(nn.Module):
    """Inverted ResNet block for MobileNet V3."""

    expansion: float
    filters: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    se_ratio: float
    block_id: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        inputs = x
        in_filters = x.shape[-1]

        se_bolock = partial(
            SeBlock,
            conv=self.conv,
        )

        prefix = "block_{:02}_".format(self.block_id)

        if self.block_id != 1:
            # Expand
            x = self.conv(
                features=int(in_filters * self.expansion),
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
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="SAME",
            feature_group_count=dw_filters,
            name=prefix + "depthwise",
        )(x)
        x = self.norm(name=prefix + "depthwise_bn")(x)
        x = self.act(x)

        if self.se_ratio:
            x = se_bolock(
                filters=_make_divisible(in_filters * self.expansion),
                se_ratio=self.se_ratio,
            )(x)

        # Project
        x = self.conv(
            features=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            name=prefix + "project",
        )(x)
        x = self.norm(name=prefix + "project_bn")(x)

        if in_filters == self.filters and self.strides == (1, 1):
            x = x + inputs

        return x


class DepthwiseSeparable(nn.Module):
    """DepthwiseSeparableConv"""

    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    out_features: int
    depth_multiplier: float = 1.0
    alpha: float = 1.0
    dw_kernel_size: Tuple[int, int] = (3, 3)
    pw_kernel_size: Tuple[int, int] = (1, 1)
    strides: Tuple[int, int] = (1, 1)
    dilation: Tuple[int, int] = (1, 1)
    pad_type: str = "SAME"

    @nn.compact
    def __call__(self, x):
        in_features = int(x.shape[-1] * self.depth_multiplier)

        x = self.conv(
            in_features,
            self.dw_kernel_size,
            strides=self.strides,
            kernel_dilation=self.dilation,
            feature_group_count=in_features,
            padding=self.pad_type,
            name="dw_conv",
        )(x)
        x = self.norm(name="dw_bn")(x)
        x = self.act(x)

        x = self.conv(
            int(self.out_features * self.alpha),
            self.pw_kernel_size,
            padding=self.pad_type,
            name="pw_conv",
        )(x)
        x = self.norm(name="pw_bn")(x)
        x = self.act(x)

        return x


class SeBlock(nn.Module):
    filters: int
    se_ratio: int
    conv: ModuleDef

    @nn.compact
    def __call__(self, x):
        inputs = x
        filters = _make_divisible(self.filters * self.se_ratio)
        in_filters = x.shape[-1]

        x = jnp.mean(x, axis=(1, 2), keepdims=True)
        x = x.reshape(-1, 1, 1, in_filters)

        x = self.conv(filters, kernel_size=(1, 1), padding="same")(x)
        x = nn.relu(x)

        x = self.conv(self.filters, kernel_size=(1, 1), padding="same")(x)
        x = jnn.hard_sigmoid(x)

        return jnp.multiply(inputs, x)
