#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from typing import Any, Callable, Dict

import jax.numpy as jnp
from flax import linen as nn

from common_layer import ResidualBlockV2

ModuleDef = Any


""" Fully Convolutional Networks for Semantic Segmentation models for Flax.

    Reference:
        Fully Convolutional Networks for Semantic Segmentation
        https://arxiv.org/abs/1411.4038
"""


class ResNetV2Backbone(nn.Module):
    """ResNet V2."""

    conv: ModuleDef
    norm: ModuleDef
    layers: Dict
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        residual_block_v2 = partial(
            ResidualBlockV2, conv=self.conv, norm=self.norm, act=self.act
        )
        return_layers = []

        x = self.conv(
            64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=[(3, 3), (3, 3)],
            name="layer_1_conv",
        )(x)

        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

        return_layers_name = ["layer_3", "layer_4", "layer_5"]

        for layer_id, layer_itmes in self.layers.items():
            x = residual_block_v2(
                filters=layer_itmes["filters"],
                strides=(1, 1),
                is_conv_shortcut=True,
                name=layer_id + "_block1",
            )(x)

            for i in range(2, layer_itmes["blocks"]):
                x = residual_block_v2(
                    filters=layer_itmes["filters"],
                    strides=(1, 1),
                    name=layer_id + "_block" + str(i),
                )(x)

            x = residual_block_v2(
                filters=layer_itmes["filters"],
                strides=layer_itmes["stride"],
                name=layer_id + "_block" + str(layer_itmes["blocks"]),
            )(x)

            if layer_id in return_layers_name:
                return_layers.append(x)

        return return_layers


ResNet50V2_layer = {
    "layer_2": {"filters": 64, "blocks": 3, "stride": (1, 1)},
    "layer_3": {"filters": 128, "blocks": 4, "stride": (2, 2)},
    "layer_4": {"filters": 256, "blocks": 6, "stride": (2, 2)},
    "layer_5": {"filters": 512, "blocks": 3, "stride": (2, 2)},
}


class FCNHead(nn.Module):
    """FCN Head Module."""

    transpose_conv: ModuleDef = nn.ConvTranspose
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, layer_3, layer_4, layer_5):
        # layer5 -> 2x upsampled prediction
        channel = layer_5.shape[-1]
        layer_5 = self.transpose_conv(
            channel // 2, (3, 3), strides=(2, 2), padding="SAME"
        )(layer_5)
        layer_5 = self.norm()(layer_5)
        layer_5 = self.act(layer_5)

        # layer 4 + 5 -> 2x upsampled prediction
        channel = layer_4.shape[-1]
        layer_4 = layer_4 + layer_5
        layer_4 = self.transpose_conv(
            channel // 2, (3, 3), strides=(2, 2), padding="SAME"
        )(layer_4)
        layer_4 = self.norm()(layer_4)
        layer_4 = self.act(layer_4)

        # layer 3 + (4 + 5) -> 8x upsampled prediction
        channel = layer_3.shape[-1]
        layer_3 = layer_3 + layer_4
        layer_3 = self.transpose_conv(
            channel // 2, (3, 3), strides=(8, 8), padding="SAME"
        )(layer_3)
        layer_3 = self.norm()(layer_3)
        layer_3 = self.act(layer_3)

        return layer_3


class FCN(nn.Module):
    """FCN Segmentation."""

    layers: Dict
    num_classes: int
    backbone: ModuleDef
    conv: ModuleDef = nn.Conv
    transpose_conv: ModuleDef = nn.ConvTranspose
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        transpose_conv = partial(self.transpose_conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            self.norm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        backbone = partial(
            self.backbone,
            layers=self.layers,
            conv=conv,
            norm=norm,
            act=self.act,
            dtype=self.dtype,
        )

        head = partial(
            FCNHead,
            transpose_conv=transpose_conv,
            norm=norm,
            act=self.act,
            dtype=self.dtype,
        )

        # backbone
        layer_3, layer_4, layer_5 = backbone()(x)

        # haed
        x = head()(layer_3, layer_4, layer_5)

        # classifier
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = conv(self.num_classes, kernel_size=(1, 1))(x)

        return jnp.asarray(x, self.dtype)
