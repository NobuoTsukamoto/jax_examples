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

from .common_layer import InvertedResBlock, _make_divisible

ModuleDef = Any

""" MobileNet v2 models for Flax.
    Reference:
        MobileNetV2: Inverted Residuals and Linear Bottlenecks
        https://arxiv.org/abs/1801.04381

        https://github.com/keras-team/keras-applications/blob/06fbeb0f16e1304f239b2296578d1c50b15a983a/keras_applications/mobilenet_v2.py
"""


class MobileNetV2Backbone(nn.Module):
    """MobileNet V2 backbone."""

    alpha: float
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        inverted_res_block = partial(
            InvertedResBlock, conv=self.conv, norm=self.norm, act=self.act
        )

        first_block_filters = _make_divisible(32 * self.alpha, 8)

        x = self.conv(
            first_block_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)
        x = self.norm()(x)
        x = self.act(x)

        x = inverted_res_block(
            filters=16, alpha=self.alpha, strides=(1, 1), expansion=1, block_id=0
        )(x)
        x = inverted_res_block(
            filters=24, alpha=self.alpha, strides=(2, 2), expansion=6, block_id=1
        )(x)
        x = inverted_res_block(
            filters=24, alpha=self.alpha, strides=(1, 1), expansion=6, block_id=2
        )(x)

        x = inverted_res_block(
            filters=32, alpha=self.alpha, strides=(2, 2), expansion=6, block_id=3
        )(x)
        x = inverted_res_block(
            filters=32, alpha=self.alpha, strides=(1, 1), expansion=6, block_id=4
        )(x)
        x = inverted_res_block(
            filters=32, alpha=self.alpha, strides=(1, 1), expansion=6, block_id=5
        )(x)

        x = inverted_res_block(
            filters=64, alpha=self.alpha, strides=(2, 2), expansion=6, block_id=6
        )(x)
        x = inverted_res_block(
            filters=64, alpha=self.alpha, strides=(1, 1), expansion=6, block_id=7
        )(x)
        x = inverted_res_block(
            filters=64, alpha=self.alpha, strides=(1, 1), expansion=6, block_id=8
        )(x)
        x = inverted_res_block(
            filters=64, alpha=self.alpha, strides=(1, 1), expansion=6, block_id=9
        )(x)

        x = inverted_res_block(
            filters=96, alpha=self.alpha, strides=(1, 1), expansion=6, block_id=10
        )(x)
        x = inverted_res_block(
            filters=96, alpha=self.alpha, strides=(1, 1), expansion=6, block_id=11
        )(x)
        x = inverted_res_block(
            filters=96, alpha=self.alpha, strides=(1, 1), expansion=6, block_id=12
        )(x)

        x = inverted_res_block(
            filters=160, alpha=self.alpha, strides=(2, 2), expansion=6, block_id=13
        )(x)
        x = inverted_res_block(
            filters=160, alpha=self.alpha, strides=(1, 1), expansion=6, block_id=14
        )(x)
        x = inverted_res_block(
            filters=160, alpha=self.alpha, strides=(1, 1), expansion=6, block_id=15
        )(x)

        x = inverted_res_block(
            filters=320, alpha=self.alpha, strides=(1, 1), expansion=6, block_id=16
        )(x)

        if self.alpha > 1.0:
            last_block_filters = _make_divisible(1280 * self.alpha, 8)
        else:
            last_block_filters = 1280

        x = self.conv(last_block_filters, kernel_size=(1, 1), name="conv_1")(x)
        x = self.norm(name="conv_1_bn")(x)
        x = self.act(x)

        return x


class MobileNetV2(nn.Module):
    """MobileNet V2."""

    alpha: float
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
            MobileNetV2Backbone,
            alpha=self.alpha,
            conv=conv,
            norm=norm,
            act=self.act,
        )

        x = backbone()(x)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x
