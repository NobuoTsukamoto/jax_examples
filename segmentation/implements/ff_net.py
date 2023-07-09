#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from typing import Any, Callable, Dict, OrderedDict

import jax
import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any

""" FFNet models for Flax.
    Reference:
        Simple and Efficient Architectures for Semantic Segmentation
        https://arxiv.org/abs/2206.08236

        https://github.com/Qualcomm-AI-research/FFNet
"""


class Stem(nn.Module):
    """Stem and backbone/encoder Module."""

    stem_layers: Dict
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for layer_id, layer_itmes in self.stem_layers.items():
            if "conv" in layer_id:
                x = self.conv(
                    features=layer_itmes["features"],
                    kernel_size=layer_itmes["kernel"],
                    strides=layer_itmes["strides"],
                    padding="SAME",
                    name=layer_id,
                )(x)
                x = self.norm(name=layer_id.replace("conv", "norm"))(x)
                x = self.act(x)

            elif "pool" in layer_id:
                x = nn.max_pool(
                    x,
                    layer_itmes["kernel"],
                    strides=layer_itmes["strides"],
                    padding="SAME",
                )

        return x


class UpHead(nn.Module):
    """Up Head  Module."""

    backbone_layers: Dict
    backbone_block: ModuleDef
    up_sample_filters: Dict
    resize_method: str
    mode: str
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        # backbone
        # Stage-1 strides
        backbone_layers = []
        for i, layer_itmes in self.backbone_layers.items():
            for j in range(layer_itmes["blocks"]):
                strides = (
                    layer_itmes["strides"] if i != "layer_1" and j == 0 else (1, 1)
                )
                if i == "layer_1" and j == 0 and self.mode != "GPU-Large":
                    strides = (2, 2)

                x = self.backbone_block(
                    features=layer_itmes["features"],
                    strides=strides,
                    conv=self.conv,
                    norm=self.norm,
                    act=self.act,
                )(x)

            backbone_layers.insert(0, x)

        # Up-head / Decoder
        row = None
        num_backbone_layers = len(backbone_layers)
        batch, height, width, _ = backbone_layers[-1].shape
        concat_list = []
        for i, x in enumerate(backbone_layers):
            key = "up_sample_layer_" + str(num_backbone_layers - i)
            features_1 = self.up_sample_filters[key]["features_1"]
            features_2 = self.up_sample_filters[key]["features_2"]

            x = self.conv(
                features=features_1,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
            )(x)
            x = self.norm()(x)
            x = self.act(x)

            # up sample + add
            if i > 0:
                y = row
                row = x
                y = self.conv(
                    features=features_1,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="SAME",
                )(y)
                y = self.norm()(y)
                y = self.act(y)
                y = jax.image.resize(
                    y,
                    shape=x.shape,
                    method=self.resize_method,
                )
                x = x + y
            else:
                row = x

            x = self.conv(
                features=features_2,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
            )(x)
            x = self.norm()(x)
            x = self.act(x)
            x = jax.image.resize(
                x,
                shape=(batch, height, width, x.shape[-1]),
                method=self.resize_method,
            )
            concat_list.insert(0, x)

        x = jnp.concatenate(concat_list, axis=-1)
        return x


class SegmentationHead(nn.Module):
    """Segmentation Head Module."""

    features: int
    num_classes: int
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        x = self.conv(
            self.features, kernel_size=(3, 3), strides=(1, 1), padding="SAME"
        )(x)
        x = self.norm()(x)
        x = self.act(x)

        x = self.conv(
            self.num_classes, kernel_size=(1, 1), strides=(1, 1), padding="SAME"
        )(x)
        return x


class FFNet(nn.Module):
    """FFNet."""

    stem_layers: Dict
    backbone_layers: OrderedDict
    backbone_block: ModuleDef
    up_sample_layers: Dict
    seg_head_features: int
    num_classes: int
    mode: str = "GPU-Large"  # GPU-Large, GPU-Small, Mobile
    conv: ModuleDef = nn.Conv
    act: Callable = nn.relu
    dtype: Any = jnp.float32

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

        # up sampling mode
        resize_method = "bilinear" if self.mode == "mobile" else "nearest"
        stem = partial(Stem, conv=conv, norm=norm, act=self.act)
        up_head = partial(UpHead, conv=conv, norm=norm, act=self.act)
        segmentation_head = partial(
            SegmentationHead, conv=conv, norm=norm, act=self.act
        )

        # Stem
        x = stem(stem_layers=self.stem_layers)(x)

        # Up-head
        x = up_head(
            backbone_layers=self.backbone_layers,
            backbone_block=self.backbone_block,
            up_sample_filters=self.up_sample_layers,
            resize_method=resize_method,
            mode=self.mode
        )(x)

        # Segmentation-head
        x = segmentation_head(
            features=self.seg_head_features, num_classes=self.num_classes
        )(x)

        print(x.shape)

        return jnp.asarray(x, self.dtype)


Stem_A = {
    "stem_conv_layer_1": {"features": 32, "kernel": (7, 7), "strides": (2, 2)},
    "stem_pool_layer_2": {"kernel": (3, 3), "strides": (2, 2)},
}

Stem_B = {
    "stem_conv_layer_1": {"features": 32, "kernel": (3, 3), "strides": (2, 2)},
    "stem_conv_layer_2": {"features": 64, "kernel": (3, 3), "strides": (2, 2)},
}

Stem_C = {
    "stem_conv_layer_1": {"features": 32, "kernel": (3, 3), "strides": (2, 2)},
    "stem_conv_layer_2": {"features": 64, "kernel": (3, 3), "strides": (2, 2)},
    "stem_conv_layer_3": {"features": 64, "kernel": (3, 3), "strides": (1, 1)},
}

ResNet150 = {
    "layer_1": {"features": 64, "blocks": 16, "strides": (1, 1)},
    "layer_2": {"features": 128, "blocks": 18, "strides": (2, 2)},
    "layer_3": {"features": 256, "blocks": 28, "strides": (2, 2)},
    "layer_4": {"features": 512, "blocks": 12, "strides": (2, 2)},
}
ResNet150S = {
    "layer_1": {"features": 64, "blocks": 16, "strides": (1, 1)},
    "layer_2": {"features": 128, "blocks": 18, "strides": (2, 2)},
    "layer_3": {"features": 192, "blocks": 28, "strides": (2, 2)},
    "layer_4": {"features": 320, "blocks": 12, "strides": (2, 2)},
}
ResNet101S = {
    "layer_1": {"features": 256, "blocks": 3, "strides": (1, 1)},
    "layer_2": {"features": 512, "blocks": 4, "strides": (2, 2)},
    "layer_3": {"features": 1024, "blocks": 23, "strides": (2, 2)},
    "layer_4": {"features": 2048, "blocks": 3, "strides": (2, 2)},
}
ResNet78S = {
    "layer_1": {"features": 64, "blocks": 6, "strides": (1, 1)},
    "layer_2": {"features": 128, "blocks": 12, "strides": (2, 2)},
    "layer_3": {"features": 192, "blocks": 12, "strides": (2, 2)},
    "layer_4": {"features": 320, "blocks": 8, "strides": (2, 2)},
}
ResNet122N = {
    "layer_1": {"features": 96, "blocks": 16, "strides": (2, 2)},
    "layer_2": {"features": 160, "blocks": 24, "strides": (2, 2)},
    "layer_3": {"features": 320, "blocks": 20, "strides": (2, 2)},
}
ResNet74N = {
    "layer_1": {"features": 96, "blocks": 8, "strides": (2, 2)},
    "layer_2": {"features": 160, "blocks": 12, "strides": (2, 2)},
    "layer_3": {"features": 320, "blocks": 16, "strides": (2, 2)},
}
ResNet46N = {
    "layer_1": {"features": 96, "blocks": 6, "strides": (2, 2)},
    "layer_2": {"features": 128, "blocks": 8, "strides": (2, 2)},
    "layer_3": {"features": 320, "blocks": 8, "strides": (2, 2)},
}
ResNet122NS = {
    "layer_1": {"features": 64, "blocks": 16, "strides": (2, 2)},
    "layer_2": {"features": 128, "blocks": 24, "strides": (2, 2)},
    "layer_3": {"features": 256, "blocks": 20, "strides": (2, 2)},
}
ResNet74NS = {
    "layer_1": {"features": 64, "blocks": 8, "strides": (2, 2)},
    "layer_2": {"features": 128, "blocks": 12, "strides": (2, 2)},
    "layer_3": {"features": 256, "blocks": 16, "strides": (2, 2)},
}
ResNet46NS = {
    "layer_1": {"features": 64, "blocks": 6, "strides": (2, 2)},
    "layer_2": {"features": 128, "blocks": 8, "strides": (2, 2)},
    "layer_3": {"features": 256, "blocks": 8, "strides": (2, 2)},
}


Up_A = {
    "up_sample_layer_1": {"features_1": 64, "features_2": 128},
    "up_sample_layer_2": {"features_1": 128, "features_2": 128},
    "up_sample_layer_3": {"features_1": 256, "features_2": 128},
    "up_sample_layer_4": {"features_1": 512, "features_2": 128},
}
Up_B = {
    "up_sample_layer_1": {"features_1": 64, "features_2": 96},
    "up_sample_layer_2": {"features_1": 128, "features_2": 96},
    "up_sample_layer_3": {"features_1": 128, "features_2": 64},
    "up_sample_layer_4": {"features_1": 256, "features_2": 32},
}
Up_C = {
    "up_sample_layer_1": {"features_1": 64, "features_2": 128},
    "up_sample_layer_2": {"features_1": 64, "features_2": 16},
    "up_sample_layer_3": {"features_1": 64, "features_2": 16},
    "up_sample_layer_4": {"features_1": 64, "features_2": 16},
}

Seg_A = 512
Seg_B = 256
Seg_C = 128
