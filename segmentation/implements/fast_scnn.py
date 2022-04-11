#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from dataclasses import field
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .common_layer import InvertedResBlock

ModuleDef = Any

""" Fast-SCNN models for Flax.
    Reference:
        Fast-SCNN: Fast Semantic Segmentation Network
        https://arxiv.org/abs/1902.04502

        https://github.com/Tramac/Fast-SCNN-pytorch
"""


class DepthwiseSeparable(nn.Module):
    """DepthwiseSeparableConv"""

    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    out_features: int
    dw_kernel_size: Tuple[int, int] = (3, 3)
    pw_kernel_size: Tuple[int, int] = (1, 1)
    strides: Tuple[int, int] = (1, 1)
    dilation: Tuple[int, int] = (1, 1)
    pad_type: str = "SAME"

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]

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
            self.out_features,
            self.pw_kernel_size,
            padding=self.pad_type,
            name="pw_conv",
        )(x)
        x = self.norm(name="pw_bn")(x)
        x = self.act(x)

        return x


class PyramidPooling(nn.Module):
    """Pyramid Pooling Module."""

    bin_sizes: Sequence[int] = field(default_factory=lambda: [1, 2, 3, 6])
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        batch, height, width, channels = x.shape
        filters = channels // len(self.bin_sizes)
        concat_list = [x]
        inputs = x

        for bin_size in self.bin_sizes:
            x = nn.avg_pool(
                inputs, window_shape=(bin_size, bin_size), strides=(bin_size, bin_size)
            )
            x = self.conv(filters, kernel_size=(1, 1))(x)
            x = self.norm()(x)
            x = self.act(x)
            x = jax.image.resize(
                x, shape=(batch, height, width, channels), method="bilinear"
            )
            concat_list.append(x)

        x = jnp.concatenate(concat_list)
        x = self.conv(filters, (1, 1))(x)
        x = self.norm()(x)
        return self.act(x)


class FeatureFusion(nn.Module):
    """Feature Fusion Module."""

    depthwise_separable_conv: ModuleDef
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x, y):
        x = self.conv(128, (1, 1))(x)
        x = self.norm()(x)

        batch, x_height, x_width, channels = x.shape
        _, y_height, y_width, _ = y.shape
        y = jax.image.resize(
            y, shape=(batch, x_height, x_width, channels), method="bilinear"
        )
        y = self.depthwise_separable_conv(
            128,
            dilation=(x_height // y_height, x_width // y_width),
        )(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(128, kernel_size=(1, 1))(y)
        y = self.norm()(y)

        x = x + y
        x = self.act(x)
        return x


class FastSCNN(nn.Module):
    """Fast-SCNN."""

    num_classes: int
    dtype: Any = jnp.float32
    conv: ModuleDef = nn.Conv
    act: Callable = nn.relu
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        depthwise_separable_conv = partial(DepthwiseSeparable, conv, norm, self.act)
        bottlenck = partial(
            InvertedResBlock, alpha=1.0, conv=conv, norm=norm, act=self.act
        )
        pyramid_pooling = partial(PyramidPooling, conv=conv, norm=norm, act=self.act)
        feature_fusion = partial(
            FeatureFusion,
            depthwise_separable_conv=depthwise_separable_conv,
            conv=conv,
            norm=norm,
            act=self.act,
        )

        batch, height, width, _ = x.shape

        # Learning to Down-sample
        x = conv(32, kernel_size=(3, 3), strides=(2, 2), name="conv_init")(x)
        x = norm()(x)
        x = self.act(x)
        x = depthwise_separable_conv(48, strides=(2, 2))(x)
        x = depthwise_separable_conv(64, strides=(2, 2))(x)

        # Global Feature Extractor
        y = x

        x = bottlenck(filters=64, strides=(2, 2), expansion=6, block_id=0)(x)
        x = bottlenck(filters=64, strides=(1, 1), expansion=6, block_id=1)(x)
        x = bottlenck(filters=64, strides=(1, 1), expansion=6, block_id=2)(x)

        x = bottlenck(filters=96, strides=(2, 2), expansion=6, block_id=3)(x)
        x = bottlenck(filters=96, strides=(1, 1), expansion=6, block_id=4)(x)
        x = bottlenck(filters=96, strides=(1, 1), expansion=6, block_id=5)(x)

        x = bottlenck(filters=128, strides=(1, 1), expansion=6, block_id=6)(x)
        x = bottlenck(filters=128, strides=(1, 1), expansion=6, block_id=7)(x)
        x = bottlenck(filters=128, strides=(1, 1), expansion=6, block_id=8)(x)

        x = pyramid_pooling()(x)

        # Feature Fusion
        x = feature_fusion()(y, x)  # high res, low res

        # Classifier
        x = depthwise_separable_conv(128, strides=(1, 1))(x)
        x = depthwise_separable_conv(128, strides=(1, 1))(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = conv(self.num_classes, kernel_size=(1, 1))(x)

        x = jax.image.resize(
            x, shape=(batch, height, width, self.num_classes), method="bilinear"
        )

        return x
