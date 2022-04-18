#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from typing import Any, Callable, Dict

import jax.numpy as jnp
import jax.nn as jnn
from flax import linen as nn

from .common_layer import _make_divisible, InvertedResBlockMobileNetV3

ModuleDef = Any

""" MobileNet V3 models for Flax.

    Reference:
        Searching for MobileNetV3
        https://arxiv.org/abs/1905.02244

        https://github.com/keras-team/keras-applications/blob/06fbeb0f16e1304f239b2296578d1c50b15a983a/keras_applications/mobilenet_v3.py
"""


class MobileNetV3(nn.Module):
    """MobileNet V3."""

    alpha: float
    num_classes: int
    layers: Dict
    last_block_filters: int
    dtype: Any = jnp.float32
    relu: Callable = nn.relu
    h_swish: Callable = jnn.hard_swish
    conv: ModuleDef = nn.Conv

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
        inverted_res_block = partial(InvertedResBlockMobileNetV3, conv=conv, norm=norm)

        first_block_filters = _make_divisible(16 * self.alpha, 8)

        x = conv(
            first_block_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            name="conv_init",
        )(x)
        x = norm()(x)
        x = self.h_swish(x)

        for block_id, layer in self.layers.items():
            x = inverted_res_block(
                expansion=layer["exp"],
                filters=layer["filters"],
                kernel_size=layer["kernel"],
                strides=layer["strides"],
                se_ratio=layer["se_ratio"],
                act=self.h_swish if layer["h_swish"] else self.relu,
                block_id=block_id,
            )(x)

        if self.alpha > 1.0:
            filters = _make_divisible(x.shape[-1] * 6 * self.alpha, 8)
        else:
            filters = _make_divisible(x.shape[-1] * 6, 8)

        x = conv(filters, kernel_size=(1, 1), name="conv_1")(x)
        x = norm()(x)
        x = self.h_swish(x)

        x = jnp.mean(x, axis=(1, 2), keepdims=True)

        x = conv(self.last_block_filters, kernel_size=(1, 1), name="conv_2")(x)
        x = jnn.hard_swish(x)

        x = conv(self.num_classes, kernel_size=(1, 1), name="conv_3")(x)
        x = x.reshape((x.shape[0], -1))  # flatten

        return jnp.asarray(x, self.dtype)


# fmt: off
Large = {
    1: {"exp": 1, "filters": 16, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": False},
    2: {"exp": 4, "filters": 24, "kernel": (3, 3), "strides": (2, 2), "se_ratio": None, "h_swish": False},
    3: {"exp": 3, "filters": 24, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": False},
    4: {"exp": 3, "filters": 40, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25, "h_swish": False},
    5: {"exp": 3, "filters": 40, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": False},
    6: {"exp": 3, "filters": 40, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": False},
    7: {"exp": 6, "filters": 80, "kernel": (3, 3), "strides": (2, 2), "se_ratio": None, "h_swish": True},
    8: {"exp": 2.5, "filters": 80, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": True},
    9: {"exp": 2.3, "filters": 80, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": True},
    10: {"exp": 2.3, "filters": 80, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": True},
    11: {"exp": 6, "filters": 112, "kernel": (3, 3), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    12: {"exp": 6, "filters": 112, "kernel": (3, 3), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    13: {"exp": 6, "filters": 160, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25, "h_swish": True},
    14: {"exp": 6, "filters": 160, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    15: {"exp": 6, "filters": 160, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
}

Small = {
    1: {"exp": 1, "filters": 16, "kernel": (3, 3), "strides": (2, 2), "se_ratio": 0.25, "h_swish": False},
    2: {"exp": 4.5, "filters": 24, "kernel": (3, 3), "strides": (2, 2), "se_ratio": None, "h_swish": False},
    3: {"exp": 88. / 24, "filters": 24, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": False},
    4: {"exp": 4, "filters": 40, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25, "h_swish": True},
    5: {"exp": 6, "filters": 40, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    6: {"exp": 6, "filters": 40, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    7: {"exp": 3, "filters": 48, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    8: {"exp": 3, "filters": 48, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    9: {"exp": 6, "filters": 96, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25, "h_swish": True},
    10: {"exp": 6, "filters": 96, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    11: {"exp": 6, "filters": 96, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
}
# fmt: on
