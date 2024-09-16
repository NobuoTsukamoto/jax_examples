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

""" ResNet v2 models for Flax.
    Reference:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/abs/1603.05027

        https://github.com/keras-team/keras/blob/v2.12.0/keras/applications/resnet_v2.py
"""


class ResNetV2(nn.Module):
    """ResNet V2."""

    layers: Dict
    num_classes: int
    dtype: Any = jnp.float32
    act: Callable = nn.relu

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
        residual_block_v2 = partial(ResidualBlockV2, conv=conv, norm=norm, act=self.act)

        x = conv(
            64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=[(3, 3), (3, 3)],
            name="Stem_Conv",
        )(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")

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

        x = norm()(x)
        x = self.act(x)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype, name="Head")(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet50V2_layer = {
    "layer_2": {"filters": 64, "blocks": 3, "stride": (1, 1)},
    "layer_3": {"filters": 128, "blocks": 4, "stride": (2, 2)},
    "layer_4": {"filters": 256, "blocks": 6, "stride": (2, 2)},
    "layer_5": {"filters": 512, "blocks": 3, "stride": (2, 2)},
}
