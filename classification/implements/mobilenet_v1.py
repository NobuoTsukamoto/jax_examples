#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from typing import Any, Callable, Optional

import jax.numpy as jnp
from flax import linen as nn

from .common_layer import DepthwiseSeparable

ModuleDef = Any


""" MobileNet v1 models for Flax.
    Reference:
        MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
        https://arxiv.org/abs/1704.04861

        https://github.com/keras-team/keras-applications/blob/dda499735da01ab6c6f029b37dbdf35cc82db136/keras_applications/mobilenet.py
"""


class MobileNetV1Backbone(nn.Module):
    """MobileNet V1 backbone."""

    alpha: float
    depth_multiplier: float
    num_classes: int
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        depthwise_separable_conv = partial(
            DepthwiseSeparable,
            self.conv,
            self.norm,
            self.act,
            depth_multiplier=self.depth_multiplier,
            alpha=self.alpha,
        )

        first_block_filters = int(32 * self.alpha)
        x = self.conv(
            first_block_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            name="conv_init",
        )(x)
        x = self.norm()(x)
        x = self.act(x)

        x = depthwise_separable_conv(64, strides=(1, 1))(x)
        x = depthwise_separable_conv(128, strides=(2, 2))(x)
        x = depthwise_separable_conv(128, strides=(1, 1))(x)
        x = depthwise_separable_conv(256, strides=(2, 2))(x)
        x = depthwise_separable_conv(256, strides=(1, 1))(x)
        x = depthwise_separable_conv(512, strides=(2, 2))(x)
        x = depthwise_separable_conv(512, strides=(1, 1))(x)
        x = depthwise_separable_conv(512, strides=(1, 1))(x)
        x = depthwise_separable_conv(512, strides=(1, 1))(x)
        x = depthwise_separable_conv(512, strides=(1, 1))(x)
        x = depthwise_separable_conv(512, strides=(1, 1))(x)
        x = depthwise_separable_conv(1024, strides=(2, 2))(x)
        x = depthwise_separable_conv(1024, strides=(1, 1))(x)

        return x


class MobileNetV1(nn.Module):
    """MobileNet V1."""

    alpha: float
    depth_multiplier: float
    num_classes: int
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    dropout_rate: Optional[float] = 0.2
    init_stochastic_depth_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        backbone = partial(
            MobileNetV1Backbone,
            alpha=self.alpha,
            depth_multiplier=self.depth_multiplier,
            num_classes=self.num_classes,
            conv=conv,
            norm=norm,
            act=self.act,
        )

        x = backbone()(x)

        shape = (x.shape[0], 1, 1, int(1024 * self.alpha))
        x = jnp.mean(x, axis=(1, 2))
        x = x.reshape(shape)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        x = conv(self.num_classes, kernel_size=(1, 1))(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        return jnp.asarray(x, self.dtype)
