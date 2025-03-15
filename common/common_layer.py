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
        y = self.conv(
            self.features, kernel_size=(3, 3), strides=self.strides, name="Conv_0"
        )(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.features, kernel_size=(3, 3), name="Conv_1")(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.features, (1, 1), self.strides, name="Project_Conv"
            )(residual)
            residual = self.norm()(residual)

        if self.stochastic_depth_drop_rate > 0.0:
            y = self.stochastic_depth(
                stochastic_depth_drop_rate=self.stochastic_depth_drop_rate,
                name="Stochastic_Depth",
            )(y)

        return self.act(residual + y)


class ResidualBlockV2(nn.Module):
    """Residual block v2"""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = None
    is_conv_shortcut: bool = False

    @nn.compact
    def __call__(self, inputs):
        pre = self.norm()(inputs)
        pre = self.act(pre)

        if self.is_conv_shortcut:
            shortcut = self.conv(
                features=self.filters * 4,
                kernel_size=(1, 1),
                strides=self.strides,
                padding="SAME",
                name="Conv_Shortcut",
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
            name="Conv_1",
        )(pre)
        x = self.norm()(x)
        x = self.act(x)

        x = self.conv(
            features=self.filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            name="Conv_2",
        )(x)
        x = self.norm()(x)
        x = self.act(x)

        x = self.conv(
            features=self.filters * 4,
            kernel_size=(1, 1),
            strides=self.strides,
            padding="SAME",
            name="Conv_3",
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
        y = self.conv(self.features, (1, 1), name="Conv_0")(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.features, (3, 3), self.strides, name="Conv_1")(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.features * 4, (1, 1), name="Conv_2")(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.features * 4, (1, 1), self.strides, name="Project_Conv"
            )(residual)
            residual = self.norm()(residual)

        if self.stochastic_depth_drop_rate > 0.0:
            y = self.stochastic_depth(
                stochastic_depth_drop_rate=self.stochastic_depth_drop_rate,
                name="Stochastic_Depth",
            )(y)
        return self.act(residual + y)


class BottleneckConvNeXtBlock(nn.Module):
    """Bottleneck ConvNeXt block."""

    features: int
    conv: ModuleDef
    linear: ModuleDef
    norm: ModuleDef
    stochastic_depth: ModuleDef
    layer_scale: ModuleDef
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
            name="DepthWise_Conv_0",
        )(x)
        y = self.norm()(y)

        # y = self.conv(self.features * 4, (1, 1))(y)
        y = self.linear(self.features * 4, name="Conv_1")(y)
        y = self.act(y)

        # y = self.conv(self.features, (1, 1))(y)
        y = self.linear(self.features, name="Conv_2")(y)

        y = self.layer_scale(projection_dim=self.features)(y)
        if self.stochastic_depth_drop_rate > 0.0:
            y = self.stochastic_depth(
                stochastic_depth_drop_rate=self.stochastic_depth_drop_rate,
            )(y)
        else:
            lambda y: y

        return residual + y


class InvertedResBlock(nn.Module):
    """Inverted ResNet block."""

    expansion: int
    strides: Tuple[int, int]
    alpha: float
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    use_expand: Optional[bool] = True
    stochastic_depth: Optional[ModuleDef] = None
    stochastic_depth_drop_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, inputs):
        in_channels = inputs.shape[-1]
        pointwise_conv_filters = int(self.filters * self.alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

        # Expand
        if self.use_expand:
            x = self.conv(
                features=in_channels * self.expansion,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                name="Expand_Conv",
            )(inputs)
            x = self.norm()(x)
            x = self.act(x)
        else:
            x = inputs

        # Depthwise
        dw_filters = x.shape[-1]
        x = self.conv(
            features=dw_filters,
            kernel_size=(3, 3),
            strides=self.strides,
            padding="SAME" if self.strides == (1, 1) else "CIRCULAR",
            feature_group_count=dw_filters,
            name="DepthWise_Conv",
        )(x)
        x = self.norm()(x)
        x = self.act(x)

        # Project
        x = self.conv(
            features=pointwise_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            name="Project_Conv",
        )(x)
        x = self.norm()(x)

        if in_channels == pointwise_filters and self.strides == (1, 1):
            if (
                self.stochastic_depth_drop_rate > 0.0
                and self.stochastic_depth is not None
            ):
                x = self.stochastic_depth(
                    stochastic_depth_drop_rate=self.stochastic_depth_drop_rate,
                )(x)
            x = x + inputs

        return x


class InvertedResBlockMobileNetV3(nn.Module):
    """Inverted ResNet block for MobileNet V3."""

    expansion: float
    filters: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    se_ratio: float
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    dtype: Any = jnp.float32
    stochastic_depth: Optional[ModuleDef] = None
    stochastic_depth_drop_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, x):
        inputs = x
        in_filters = x.shape[-1]

        se_block = partial(
            SeBlock,
            conv=partial(nn.Conv, use_bias=True, dtype=self.dtype),
            act1=nn.relu,
            act2=nn.hard_sigmoid,
        )

        if self.expansion > 1.0:
            # Expand
            x = self.conv(
                features=_make_divisible(in_filters * self.expansion),
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                name="Expand_Conv",
            )(x)
            x = self.norm()(x)
            x = self.act(x)

        # Depthwise
        dw_filters = x.shape[-1]
        x = self.conv(
            features=dw_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="SAME",
            feature_group_count=dw_filters,
            name="DepthWise_Conv",
        )(x)
        x = self.norm()(x)
        x = self.act(x)

        if self.se_ratio:
            x = se_block(
                in_filters=_make_divisible(in_filters * self.expansion),
                out_filters=dw_filters,
                se_ratio=self.se_ratio,
            )(x)

        # Project
        x = self.conv(
            features=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            name="Project_Conv",
        )(x)
        x = self.norm()(x)

        if in_filters == self.filters and self.strides == (1, 1):
            if self.stochastic_depth and self.stochastic_depth_drop_rate > 0.0:
                x = self.stochastic_depth(
                    stochastic_depth_drop_rate=self.stochastic_depth_drop_rate
                )(x)
            x = x + inputs

        return x


class InvertedResBlockEfficientNet(nn.Module):
    """Inverted ResNet block for EfficientNet."""

    expansion: float
    out_filters: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    se_ratio: float
    conv: ModuleDef
    norm: ModuleDef
    act: Callable = None
    stochastic_depth: Optional[ModuleDef] = None
    stochastic_depth_drop_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, x):
        inputs = x
        in_filters = x.shape[-1]

        se_block = partial(
            SeBlock, conv=self.conv, act1=self.act, act2=jnn.sigmoid, divisor=2
        )
        features = int(in_filters * self.expansion)

        if self.expansion > 1.0:
            # Expand
            x = self.conv(
                features=features,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                name="Expand_Conv",
            )(x)
            x = self.norm()(x)
            x = self.act(x)

        # Depthwise
        dw_filters = x.shape[-1]
        x = self.conv(
            features=dw_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="SAME",
            feature_group_count=dw_filters,
            name="DepthWise_Conv",
        )(x)
        x = self.norm()(x)
        x = self.act(x)

        # Squeeze and Excitation
        if self.se_ratio:
            x = se_block(
                in_filters=in_filters,
                out_filters=dw_filters,
                se_ratio=self.se_ratio,
            )(x)

        # Project
        x = self.conv(
            features=self.out_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            name="Project_Conv",
        )(x)
        x = self.norm()(x)

        if in_filters == self.out_filters and self.strides == (1, 1):
            if self.stochastic_depth and self.stochastic_depth_drop_rate > 0.0:
                x = self.stochastic_depth(
                    stochastic_depth_drop_rate=self.stochastic_depth_drop_rate
                )(x)
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

    @nn.compact
    def __call__(self, x):
        in_features = int(x.shape[-1] * self.depth_multiplier)

        x = self.conv(
            in_features,
            self.dw_kernel_size,
            strides=self.strides,
            kernel_dilation=self.dilation,
            feature_group_count=in_features,
            padding="SAME" if self.strides == (1, 1) else "CIRCULAR",
            name="DepthWise_Conv",
        )(x)
        x = self.norm()(x)
        x = self.act(x)

        x = self.conv(
            int(self.out_features * self.alpha),
            self.pw_kernel_size,
            padding="SAME",
            name="PointWise_Conv",
        )(x)
        x = self.norm()(x)
        x = self.act(x)

        return x


class SeBlock(nn.Module):
    """Squeeze-and-Excitation Networks"""

    in_filters: int
    out_filters: int
    se_ratio: float
    conv: ModuleDef
    act1: Callable = nn.relu
    act2: Callable = nn.hard_sigmoid
    use_bias: bool = True
    divisor: int = 8
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        inputs = x
        se_filters = _make_divisible(
            self.in_filters * self.se_ratio, divisor=self.divisor
        )

        x = jnp.mean(x, axis=(1, 2), keepdims=True, dtype=self.dtype)
        x = self.conv(
            se_filters, kernel_size=(1, 1), padding="SAME", use_bias=self.use_bias
        )(x)
        x = self.act1(x)

        x = self.conv(
            self.out_filters, kernel_size=(1, 1), padding="SAME", use_bias=self.use_bias
        )(x)
        x = self.act2(x)

        return jnp.multiply(inputs, x)
