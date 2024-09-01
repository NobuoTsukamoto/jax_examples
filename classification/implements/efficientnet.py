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
from .stochastic_depth import StochasticDepth

ModuleDef = Any

""" EfficientNet models for Flax.

    Reference:
        EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
        https://arxiv.org/abs/1905.11946
"""


class EfficientNetBackbone(nn.Module):
    """EfficientNet Backbone."""

    layers: Dict
    last_block_filters: int
    dtype: Any = jnp.float32
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    stochastic_depth: ModuleDef = StochasticDepth
    act: Callable = jnn.swish
    width_coefficient: float = 1.0
    depth_coefficient: float = 1.0
    dropout_rate: Optional[float] = 0.2
    depth_divisor: int = 8

    def _round_filters(filters: int, width_coefficient: float, divisor: int):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)

        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(repeats: int, depth_coefficient: float):
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
        stem_filters = self._round_filters(32, self.alpha, 8)
        x = self.conv(
            stem_filters, kernel_size=(3, 3), strides=(2, 2), name="Stem_Conv"
        )(x)
        x = self.norm(name="Stem_Bn")(x)
        x = self.act(x)

        num_stage = 0
        blocks = float(
            sum(self._round_repeats(i["layer"]) for i in self.layers.items())
        )
        for block_id, layer in self.layers.items():
            filters = self._round_filters(layer["filters"])
            repeats = self._round_repeats(layer["layer"])

            for j in range(repeats):
                if j > 0:
                    strides = (1, 1)
                else:
                    strides = layer["strides"]

                drop_rate = self.init_stochastic_depth_rate * num_stage / blocks

                x = inverted_res_block(
                    expansion=layer["exp"],
                    filters=filters,
                    kernel_size=layer["kernel"],
                    strides=strides,
                    se_ratio=layer["se_ratio"],
                    act=self.act,
                    stochastic_depth_drop_rate=drop_rate,
                    block_id=int(block_id),
                )(x)
                num_stage += 1

        x = self.conv(self.last_block_filters, kernel_size=(1, 1), name="conv_2")(x)
        x = self.norm()(x)
        x = self.act(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2), keepdims=True)

        return x


class EfficientNet(nn.Module):
    """EfficientNet."""

    dtype: Any = jnp.float32
    dropout_rate: Optional[float] = 0.2
    init_stochastic_depth_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):

        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.997,
            epsilon=0.001,
            dtype=self.dtype,
        )
        backbone = partial(
            EfficientNetBackbone,
            alpha=self.alpha,
            layers=self.layers,
            dtype=self.dtype,
            conv=conv,
            norm=norm,
        )

        x = backbone()(x)

        # Dropout
        if self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        x = conv(self.num_classes, kernel_size=(1, 1), name="head")(x)
        x = x.reshape((x.shape[0], -1))  # flatten

        return jnp.asarray(x, self.dtype)


# fmt: off
Large = {
    "2": {"layers": 1, "exp": 1, "filters":  16, "kernel": (3, 3), "strides": (1, 1), "se_ratio": 0.25},
    "3": {"layers": 2, "exp": 6, "filters":  24, "kernel": (3, 3), "strides": (2, 2), "se_ratio": 0.25},
    "4": {"layers": 2, "exp": 6, "filters":  40, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25},
    "5": {"layers": 3, "exp": 6, "filters":  80, "kernel": (3, 3), "strides": (2, 2), "se_ratio": 0.25},
    "6": {"layers": 3, "exp": 6, "filters": 112, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25},
    "7": {"layers": 4, "exp": 6, "filters": 192, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25},
    "8": {"layers": 1, "exp": 6, "filters": 320, "kernel": (3, 3), "strides": (1, 1), "se_ratio": 0.25},
}
# fmt: on
