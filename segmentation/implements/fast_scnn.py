#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from dataclasses import field
from functools import partial
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from common_layer import InvertedResBlock, DepthwiseSeparable

ModuleDef = Any

""" Fast-SCNN models for Flax.
    Reference:
        Fast-SCNN: Fast Semantic Segmentation Network
        https://arxiv.org/abs/1902.04502

        https://github.com/Tramac/Fast-SCNN-pytorch
"""


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

        for bin_size in self.bin_sizes:
            y = nn.avg_pool(
                x, window_shape=(bin_size, bin_size), strides=(bin_size, bin_size)
            )
            y = self.conv(filters, kernel_size=(1, 1))(y)
            y = self.norm()(y)
            y = self.act(y)
            y = jax.image.resize(
                y, shape=(batch, height, width, filters), method="bilinear"
            )
            concat_list.append(y)

        x = jnp.concatenate(concat_list, axis=-1)
        x = self.conv(channels, kernel_size=(1, 1))(x)
        x = self.norm()(x)
        return self.act(x)


class FeatureFusion(nn.Module):
    """Feature Fusion Module."""

    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, high_res, low_res):
        high_res = self.conv(128, kernel_size=(1, 1))(high_res)
        high_res = self.norm()(high_res)

        _, x_height, x_width, _ = high_res.shape
        batch, _, _, channels = low_res.shape
        low_res = jax.image.resize(
            low_res, shape=(batch, x_height, x_width, channels), method="bilinear"
        )
        # DepthwiseConv
        dw_filters = low_res.shape[-1]
        low_res = self.conv(
            features=dw_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            feature_group_count=dw_filters,
        )(low_res)
        low_res = self.norm()(low_res)
        low_res = self.act(low_res)
        low_res = self.conv(128, kernel_size=(1, 1))(low_res)
        low_res = self.norm()(low_res)

        x = high_res + low_res
        return self.act(x)


class FastSCNN(nn.Module):
    """Fast-SCNN."""

    num_classes: int
    output_size: tuple[int, int]
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
            conv=conv,
            norm=norm,
            act=self.act,
        )

        output_shape = (
            x.shape[0],
            self.output_size[0],
            self.output_size[1],
            self.num_classes,
        )

        # Learning to Down-sample
        x = conv(32, kernel_size=(3, 3), strides=(2, 2), name="conv_init")(x)
        x = norm()(x)
        x = self.act(x)
        x = depthwise_separable_conv(48, strides=(2, 2))(x)
        high_res = depthwise_separable_conv(64, strides=(2, 2))(x)

        # aux loss
        aux_loss = conv(self.num_classes, kernel_size=(1, 1))(high_res)
        aux_loss = jax.image.resize(
            aux_loss,
            shape=output_shape,
            method="bilinear",
        )
        aux_loss = jnp.asarray(aux_loss, self.dtype)

        # Global Feature Extractor
        x = bottlenck(filters=64, strides=(2, 2), expansion=6, block_id=0)(high_res)
        x = bottlenck(filters=64, strides=(1, 1), expansion=6, block_id=1)(x)
        x = bottlenck(filters=64, strides=(1, 1), expansion=6, block_id=2)(x)

        x = bottlenck(filters=96, strides=(2, 2), expansion=6, block_id=3)(x)
        x = bottlenck(filters=96, strides=(1, 1), expansion=6, block_id=4)(x)
        x = bottlenck(filters=96, strides=(1, 1), expansion=6, block_id=5)(x)

        x = bottlenck(filters=128, strides=(1, 1), expansion=6, block_id=6)(x)
        x = bottlenck(filters=128, strides=(1, 1), expansion=6, block_id=7)(x)
        x = bottlenck(filters=128, strides=(1, 1), expansion=6, block_id=8)(x)

        # Pyramid Pooing
        low_res = pyramid_pooling()(x)

        # Feature Fusion
        x = feature_fusion()(high_res, low_res)  # high res, low res

        # Classifier
        x = depthwise_separable_conv(128, strides=(1, 1))(x)
        x = depthwise_separable_conv(128, strides=(1, 1))(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = conv(self.num_classes, kernel_size=(1, 1))(x)

        x = jax.image.resize(
            x,
            shape=output_shape,
            method="bilinear",
        )
        x = jnp.asarray(x, self.dtype)
        return {"output": x, "aux_loss": aux_loss}
