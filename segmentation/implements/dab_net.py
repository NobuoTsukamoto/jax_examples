#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn


ModuleDef = Any

""" DABNet  models for Flax.
    Reference:
        DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation
        https://arxiv.org/abs/1907.11357

        https://github.com/Reagan1311/DABNet
"""


class DepthwiseAsymmetricBottleneck(nn.Module):
    """DAB Module."""

    in_features: int
    dilation: int
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.PReLU

    @nn.compact
    def __call__(self, x):
        dab_features = self.in_features // 2

        y = self.norm()(x)
        y = self.act()(y)

        y = self.conv(dab_features, kernel_size=(3, 3), strides=(1, 1))(y)
        y = self.norm()(y)
        y = self.act()(y)

        y1 = self.conv(
            dab_features,
            kernel_size=(3, 1),
            feature_group_count=dab_features,
            strides=(1, 1),
            padding="SAME",
        )(y)
        y1 = self.norm()(y1)
        y1 = self.act()(y1)
        y1 = self.conv(
            dab_features,
            kernel_size=(1, 3),
            feature_group_count=dab_features,
            strides=(1, 1),
            padding="SAME",
        )(y1)
        y1 = self.norm()(y1)
        y1 = self.act()(y1)

        y2 = self.conv(
            dab_features,
            kernel_size=(3, 1),
            kernel_dilation=(self.dilation, 1),
            feature_group_count=dab_features,
            strides=(1, 1),
            padding="SAME",
        )(y)
        y2 = self.norm()(y2)
        y2 = self.act()(y2)
        y2 = self.conv(
            dab_features,
            kernel_size=(1, 3),
            kernel_dilation=(1, self.dilation),
            feature_group_count=dab_features,
            strides=(1, 1),
            padding="SAME",
        )(y2)

        y = y1 + y2
        y = self.norm()(y)
        y = self.act()(y)

        y = self.conv(
            self.in_features,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
        )(y)

        return x + y


class DABNet(nn.Module):
    """DABNet."""

    num_classes: int
    output_size: tuple[int, int]
    dtype: Any = jnp.float32
    conv: ModuleDef = nn.Conv
    act: Callable = nn.PReLU

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
        dab_block = partial(
            DepthwiseAsymmetricBottleneck, conv=conv, norm=norm, act=self.act
        )

        output_shape = (
            x.shape[0],
            self.output_size[0],
            self.output_size[1],
            self.num_classes,
        )

        # DownSampling
        x1 = nn.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        x2 = nn.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x2 = nn.avg_pool(x2, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        x3 = nn.avg_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x3 = nn.avg_pool(x3, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x3 = nn.avg_pool(x3, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # Inital Block
        y = conv(32, kernel_size=(3, 3), strides=(2, 2), name="conv_layer_1")(x)
        y = norm()(y)
        y = self.act()(y)
        y = conv(32, kernel_size=(3, 3), strides=(1, 1), name="conv_layer_2")(y)
        y = norm()(y)
        y = self.act()(y)
        y = conv(32, kernel_size=(3, 3), strides=(1, 1), name="conv_layer_3")(y)
        y = norm()(y)
        y = self.act()(y)

        y = jnp.concatenate([y, x1], axis=-1)
        y = norm()(y)
        y = self.act()(y)

        # DownSampling 1
        y_conv = conv(64 - y.shape[-1], kernel_size=(3, 3), strides=(2, 2))(y)
        y_max_pool = nn.max_pool(y, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        y = jnp.concatenate([y_conv, y_max_pool], axis=-1)
        y = norm()(y)
        down_sample_1 = self.act()(y)

        # DAB 1
        y = dab_block(in_features=64, dilation=2)(down_sample_1)
        y = dab_block(in_features=64, dilation=2)(y)
        y = dab_block(in_features=64, dilation=2)(y)

        y = jnp.concatenate([y, down_sample_1, x2], axis=-1)
        y = norm()(y)
        y = self.act()(y)

        # DownSampling 2
        y = conv(128, kernel_size=(3, 3), strides=(2, 2), name="downsample_conv_2")(y)
        y = norm()(y)
        down_sample_2 = self.act()(y)

        # DAB 2
        y = dab_block(in_features=128, dilation=4)(down_sample_2)
        y = dab_block(in_features=128, dilation=4)(y)
        y = dab_block(in_features=128, dilation=8)(y)
        y = dab_block(in_features=128, dilation=8)(y)
        y = dab_block(in_features=128, dilation=16)(y)
        y = dab_block(in_features=128, dilation=16)(y)

        y = jnp.concatenate([y, down_sample_2, x3], axis=-1)
        y = norm()(y)
        y = self.act()(y)

        # Classfier
        y = conv(self.num_classes, kernel_size=(1, 1))(y)

        y = jax.image.resize(
            y,
            shape=output_shape,
            method="bilinear",
        )
        y = jnp.asarray(y, self.dtype)
        return {
            "output": y,
        }
