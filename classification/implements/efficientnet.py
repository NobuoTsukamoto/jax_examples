#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import math

from functools import partial
from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp
import jax.nn as jnn
from flax import linen as nn

from common_layer import InvertedResBlockEfficientNet
from stochastic_depth import StochasticDepth

ModuleDef = Any

""" EfficientNet models for Flax.

    Reference:
        EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
        https://arxiv.org/abs/1905.11946
        https://github.com/keras-team/keras/blob/v3.5.0/keras/src/applications/efficientnet.py
        https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
        https://github.com/tensorflow/models/tree/master/official/vision
"""

# fmt: off
BLOCK = {
    "2": {"kernel": (3, 3), "layers": 1, "out_filters":  16, "exp": 1, "id_skip": True, "strides": (1, 1), "se_ratio": 0.25},
    "3": {"kernel": (3, 3), "layers": 2, "out_filters":  24, "exp": 6, "id_skip": True, "strides": (2, 2), "se_ratio": 0.25},
    "4": {"kernel": (5, 5), "layers": 2, "out_filters":  40, "exp": 6, "id_skip": True, "strides": (2, 2), "se_ratio": 0.25},
    "5": {"kernel": (3, 3), "layers": 3, "out_filters":  80, "exp": 6, "id_skip": True, "strides": (2, 2), "se_ratio": 0.25},
    "6": {"kernel": (5, 5), "layers": 3, "out_filters": 112, "exp": 6, "id_skip": True, "strides": (1, 1), "se_ratio": 0.25},
    "7": {"kernel": (5, 5), "layers": 4, "out_filters": 192, "exp": 6, "id_skip": True, "strides": (2, 2), "se_ratio": 0.25},
    "8": {"kernel": (3, 3), "layers": 1, "out_filters": 320, "exp": 6, "id_skip": True, "strides": (1, 1), "se_ratio": 0.25},
}
# fmt: on


class EfficientNetBackbone(nn.Module):
    """EfficientNet Backbone."""

    layers: Dict
    init_stochastic_depth_rate: float
    dtype: Any = jnp.float32
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    stochastic_depth: ModuleDef = StochasticDepth
    act: Callable = jnn.swish
    width_coefficient: Optional[float] = 1.0
    depth_coefficient: Optional[float] = 1.0
    dropout_rate: Optional[float] = 0.2
    last_block_filters: Optional[int] = 1280
    depth_divisor: Optional[int] = 8

    def _round_filters(self, filters: int, width_coefficient: float, divisor: int):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)

        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(self, repeats: int, depth_coefficient: float):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    @nn.compact
    def __call__(self, x):

        inverted_res_block = partial(
            InvertedResBlockEfficientNet,
            conv=self.conv,
            norm=self.norm,
            act=self.act,
            stochastic_depth=self.stochastic_depth,
        )

        # stem
        stem_filters = self._round_filters(
            32, self.width_coefficient, self.depth_divisor
        )
        x = self.conv(
            stem_filters, kernel_size=(3, 3), strides=(2, 2), name="Stem_Conv"
        )(x)
        x = self.norm()(x)
        x = self.act(x)

        num_stage = 0
        blocks = float(
            sum(
                self._round_repeats(i[1]["layers"], self.depth_coefficient)
                for i in self.layers.items()
            )
        )
        for _, layer in self.layers.items():
            out_filters = self._round_filters(
                layer["out_filters"], self.width_coefficient, self.depth_divisor
            )
            repeats = self._round_repeats(layer["layers"], self.depth_coefficient)

            for j in range(repeats):
                if j > 0:
                    strides = (1, 1)
                else:
                    strides = layer["strides"]

                drop_rate = self.init_stochastic_depth_rate * float(num_stage) / blocks

                x = inverted_res_block(
                    expansion=layer["exp"],
                    out_filters=out_filters,
                    kernel_size=layer["kernel"],
                    strides=strides,
                    se_ratio=layer["se_ratio"],
                    act=self.act,
                    stochastic_depth_drop_rate=drop_rate,
                )(x)
                num_stage += 1

        x = self.conv(self.last_block_filters, kernel_size=(1, 1))(x)
        x = self.norm()(x)
        x = self.act(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2), keepdims=True)

        return x


class EfficientNet(nn.Module):
    """EfficientNet."""

    num_classes: int
    init_stochastic_depth_rate: Optional[float] = 0.0
    width_coefficient: Optional[float] = 1.0
    depth_coefficient: Optional[float] = 1.0
    dropout_rate: Optional[float] = 0.2
    dtype: Any = jnp.float32
    use_bias: Optional[bool] = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        kernel_initializer = nn.initializers.variance_scaling(
            scale=2.0, mode="fan_out", distribution="truncated_normal"
        )
        dense_initializer = nn.initializers.variance_scaling(
            scale=1.0 / 3.0, mode="fan_out", distribution="truncated_normal"
        )

        conv = partial(
            nn.Conv,
            use_bias=self.use_bias,
            kernel_init=kernel_initializer,
            dtype=self.dtype,
        )
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.99,
            epsilon=1e-3,
            dtype=self.dtype,
        )
        stochastic_depth = partial(StochasticDepth, deterministic=not train)

        backbone = partial(
            EfficientNetBackbone,
            layers=BLOCK,
            width_coefficient=self.width_coefficient,
            depth_coefficient=self.depth_coefficient,
            dtype=self.dtype,
            conv=conv,
            norm=norm,
            stochastic_depth=stochastic_depth,
            init_stochastic_depth_rate=self.init_stochastic_depth_rate,
        )

        x = backbone()(x)

        # Dropout
        if self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        x = nn.Dense(
            self.num_classes,
            name="Head",
            kernel_init=dense_initializer,
            dtype=self.dtype
        )(x)

        return jnp.asarray(x, self.dtype)
