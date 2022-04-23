#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from implements.fast_scnn import FastSCNN
from implements.lite_raspp import LiteRASPP, MobileNetV3, Large, Small


# FastSCNN
Fast_SCNN = partial(FastSCNN)

# Lite R-ASPP MobileNet V3 Large
LRASPP_MobileNetV3_Large = partial(
    LiteRASPP, backbone=MobileNetV3, layers=Large, segmentation_head_filters=128
)

# Lite R-ASPP MobileNet V3 Large
LRASPP_MobileNetV3_Small = partial(
    LiteRASPP, backbone=MobileNetV3, layers=Small, segmentation_head_filters=128
)
