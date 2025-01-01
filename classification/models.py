#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from implements.mobilenet_v1 import MobileNetV1
from implements.mobilenet_v2 import MobileNetV2
from implements.mobilenet_v3 import MobileNetV3, Large, Small
from implements.resnet_v2 import ResNetV2, ResNet50V2_layer
from implements.resnet_v1 import ResNet
from implements.convnext import ConvNeXt
from implements.vit import ViT
from implements.efficientnet import EfficientNet

from common_layer import (
    ResNetBlock,
    BottleneckResNetBlock,
    BottleneckConvNeXtBlock,
)

# MobileNetV1 alpha=1.0 depth_multiplier=1.0
MobileNetV1_10 = partial(MobileNetV1, alpha=1.0, depth_multiplier=1.0)

# MobileNetV2
MobileNetV2_10 = partial(MobileNetV2, alpha=1.0)

# MobieNetV3 Large
MobileNetV3_Large = partial(
    MobileNetV3, alpha=1.0, layers=Large, last_block_filters=1280
)

# MobileNetV3 Small
MobileNetV3_Small = partial(
    MobileNetV3, alpha=1.0, layers=Small, last_block_filters=1024
)

# ResBet v1
ResNet18 = partial(
    ResNet,
    stage_sizes=[2, 2, 2, 2],
    num_filters=[64, 128, 256, 512],
    block_cls=ResNetBlock,
)
ResNet50 = partial(
    ResNet,
    stage_sizes=[3, 4, 6, 3],
    num_filters=[64, 128, 256, 512],
    block_cls=BottleneckResNetBlock,
)

# ResNet50 v2
ResNet50V2 = partial(ResNetV2, layers=ResNet50V2_layer)

# ConvNeXt V1
ConvNeXt_T = partial(
    ConvNeXt,
    stage_sizes=[3, 3, 9, 3],
    num_filters=[96, 192, 384, 768],
    kernel_size=(7, 7),
    block_cls=BottleneckConvNeXtBlock,
)

ConvNeXt_S = partial(
    ConvNeXt,
    stage_sizes=[3, 3, 27, 3],
    num_filters=[96, 192, 384, 768],
    kernel_size=(7, 7),
    block_cls=BottleneckConvNeXtBlock,
)
ConvNeXt_B = partial(
    ConvNeXt,
    stage_sizes=[3, 3, 27, 3],
    num_filters=[128, 256, 512, 1024],
    kernel_size=(7, 7),
    block_cls=BottleneckConvNeXtBlock,
)

# ViT-4T
ViT_4T = partial(
    ViT,
    num_patch_row=4,
    num_blocks=12,
    head=3,
    hidden_dim=192,
)

# ViT-4S
ViT_4S = partial(
    ViT,
    num_patch_row=4,
    num_blocks=12,
    head=6,
    hidden_dim=384,
)

# EfficientNet B0
EfficientNet_B0 = partial(
    EfficientNet,
    width_coefficient=1.0,
    depth_coefficient=1.0,
    dropout_rate=0.2,
)

# EfficientNet B1
EfficientNet_B1 = partial(
    EfficientNet,
    width_coefficient=1.0,
    depth_coefficient=1.1,
    dropout_rate=0.2,
)
