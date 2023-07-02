#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from implements.mobilenet_v1 import MobileNetV1
from implements.mobilenet_v2 import MobileNetV2
from implements.mobilenet_v3 import MobileNetV3, Large, Small
from implements.resnet_v2 import ResNetV2, ResNet50V2_layer

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


# ResNet50 v2
ResNet50V2 = partial(ResNetV2, layers=ResNet50V2_layer)
