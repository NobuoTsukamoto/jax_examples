#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.nn as jnn
from flax import linen as nn

from common_layer import _make_divisible, SeBlock

ModuleDef = Any


""" Lite R-ASPP MobileNet V3 models for Flax.

    Reference:
        Searching for MobileNetV3
        https://arxiv.org/abs/1905.02244
"""


class LiteRASPPHead(nn.Module):
    """Lite-ASPP Head Module."""

    num_classes: int
    aspp_convs_filters: int = 128
    image_pooling_window_shape: Tuple[int, int] = (25, 25)
    image_pooling_stride: Tuple[int, int] = (4, 5)
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, high_pos, low_pos):
        # 1x1 conv + bn + relu
        x = self.conv(self.aspp_convs_filters, kernel_size=(1, 1))(high_pos)
        x = self.norm()(x)
        x = self.act(x)

        # avg pool -> 1x1 conv -> sigmoid -> upsampling
        batch, height, width, _ = high_pos.shape
        y = nn.avg_pool(
            high_pos,
            window_shape=self.image_pooling_window_shape,
            strides=self.image_pooling_stride,
        )
        y = self.conv(self.aspp_convs_filters, kernel_size=(1, 1))(y)
        y = nn.sigmoid(y)
        y = jax.image.resize(
            y, shape=(batch, height, width, x.shape[-1]), method="bilinear"
        )

        # multiply -> upsampling -> 1x1 conv
        x = jnp.multiply(x, y)

        channel = x.shape[-1]
        batch, height, width, _ = low_pos.shape
        x = jax.image.resize(
            x, shape=(batch, height, width, channel), method="bilinear"
        )
        x = self.conv(self.num_classes, kernel_size=(1, 1))(x)
        y = self.conv(self.num_classes, kernel_size=(1, 1))(low_pos)

        return jnp.add(x, y)


class MobileNetV3(nn.Module):
    """MobileNet V3 backbone."""

    conv: ModuleDef
    norm: ModuleDef
    layers: Dict
    alpha: float = 1.0
    skip_layer: int = 6
    relu: Callable = nn.relu
    h_swish: Callable = jnn.hard_swish
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        se_bolock = partial(SeBlock, conv=self.conv)

        first_block_filters = _make_divisible(16 * self.alpha, 8)

        x = self.conv(
            first_block_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            name="conv_init",
        )(x)
        x = self.norm()(x)
        x = self.h_swish(x)

        for block_id, layer in self.layers.items():
            expansion = layer["exp"]
            filters = layer["filters"]
            kernel_size = layer["kernel"]
            strides = layer["strides"]
            se_ratio = layer["se_ratio"]
            act = self.h_swish if layer["h_swish"] else self.relu
            block_id = int(block_id)

            inputs = x
            in_filters = x.shape[-1]
            prefix = "block_{:02}_".format(block_id)

            if block_id != 0:
                # Expand
                x = self.conv(
                    features=int(in_filters * expansion),
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="SAME",
                    name=prefix + "expand_conv",
                )(x)
                x = self.norm(name=prefix + "expand_bn")(x)
                x = act(x)

            # Depthwise
            dw_filters = x.shape[-1]
            x = self.conv(
                features=dw_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="SAME",
                feature_group_count=dw_filters,
                name=prefix + "depthwise_conv",
            )(x)
            x = self.norm(name=prefix + "depthwise_bn")(x)
            x = act(x)

            if block_id == self.skip_layer:
                low_pos = x

            if se_ratio:
                x = se_bolock(
                    filters=_make_divisible(in_filters * expansion),
                    se_ratio=se_ratio,
                )(x)

            # Project
            x = self.conv(
                features=filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                name=prefix + "project",
            )(x)
            x = self.norm(name=prefix + "project_bn")(x)

            if in_filters == filters and strides == (1, 1):
                x = jnp.add(x, inputs)

        if self.alpha > 1.0:
            filters = _make_divisible(x.shape[-1] * 6 * self.alpha, 8)
        else:
            filters = _make_divisible(x.shape[-1] * 6, 8)

        x = self.conv(filters, kernel_size=(1, 1))(x)
        x = self.norm()(x)
        high_pos = self.h_swish(x)

        return high_pos, low_pos


class LiteRASPP(nn.Module):
    """Lite-ASPP Segmentation."""

    layers: Dict
    num_classes: int
    backbone: ModuleDef
    segmentation_head_filters: int = 128
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            self.norm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        head = partial(
            LiteRASPPHead,
            conv=conv,
            norm=norm,
            num_classes=self.num_classes,
            aspp_convs_filters=self.segmentation_head_filters,
            dtype=self.dtype,
        )

        batch, height, width, _ = x.shape

        high_pos, low_pos = self.backbone(
            conv=conv, norm=norm, layers=self.layers, dtype=self.dtype
        )(x)
        x = head()(high_pos, low_pos)

        channel = x.shape[-1]
        x = jax.image.resize(
            x, shape=(batch, height, width, channel), method="bilinear"
        )
        return jnp.asarray(x, self.dtype)


# fmt: off
Large = {
    "0": {"exp": 1, "filters": 16, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": False},
    "1": {"exp": 4, "filters": 24, "kernel": (3, 3), "strides": (2, 2), "se_ratio": None, "h_swish": False},
    "2": {"exp": 3, "filters": 24, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": False},
    "3": {"exp": 3, "filters": 40, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25, "h_swish": False},
    "4": {"exp": 3, "filters": 40, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": False},
    "5": {"exp": 3, "filters": 40, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": False},
    "6": {"exp": 6, "filters": 80, "kernel": (3, 3), "strides": (2, 2), "se_ratio": None, "h_swish": True},
    "7": {"exp": 2.5, "filters": 80, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": True},
    "8": {"exp": 2.3, "filters": 80, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": True},
    "9": {"exp": 2.3, "filters": 80, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": True},
    "10": {"exp": 6, "filters": 112, "kernel": (3, 3), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "11": {"exp": 6, "filters": 112, "kernel": (3, 3), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "12": {"exp": 6/2, "filters": 160//2, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25, "h_swish": True},
    "13": {"exp": 6, "filters": 160//2, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "14": {"exp": 6, "filters": 160//2, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
}

Small = {
    "0": {"exp": 1, "filters": 16, "kernel": (3, 3), "strides": (2, 2), "se_ratio": 0.25, "h_swish": False},
    "1": {"exp": 4.5, "filters": 24, "kernel": (3, 3), "strides": (2, 2), "se_ratio": None, "h_swish": False},
    "2": {"exp": 88. / 24, "filters": 24, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": False},
    "3": {"exp": 4, "filters": 40, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25, "h_swish": True},
    "4": {"exp": 6, "filters": 40, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "5": {"exp": 6, "filters": 40, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "6": {"exp": 3, "filters": 48, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "7": {"exp": 3, "filters": 48, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "8": {"exp": 6/2, "filters": 96//2, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25, "h_swish": True},
    "9": {"exp": 6, "filters": 96//2, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "10": {"exp": 6, "filters": 96//2, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
}
# fmt: on
