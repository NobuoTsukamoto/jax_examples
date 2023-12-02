#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from implements.fast_scnn import FastSCNN
from implements.lite_raspp import LiteRASPP, MobileNetV3, Large, Small
from implements.fcn import FCN, ResNet50V2_layer, ResNetV2Backbone
from implements.ff_net import (
    FFNet,
    Stem_A,
    Stem_B,
    Stem_C,
    Up_A,
    Up_B,
    Up_C,
    Seg_A,
    Seg_B,
    Seg_C,
    ResNet150,
    ResNet150S,
    ResNet101S,
    ResNet78S,
    ResNet122N,
    ResNet74N,
    ResNet46N,
    ResNet122NS,
    ResNet74NS,
    ResNet46NS,
)
from implements.dab_net import DABNet
from implements.led_net import LEDNet
from implements.common_layer import ResNetBlock

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

# FCN
FCN_ResNetV2 = partial(FCN, backbone=ResNetV2Backbone, layers=ResNet50V2_layer)

# FFNet-GPU-Large
FFNet_ResNet122N_CBB = partial(
    FFNet,
    stem_layers=Stem_C,
    backbone_layers=ResNet122N,
    backbone_block=ResNetBlock,
    up_sample_layers=Up_B,
    seg_head_features=Seg_B,
)
FFNet_ResNet74N_CBB = partial(
    FFNet,
    stem_layers=Stem_C,
    backbone_layers=ResNet74N,
    backbone_block=ResNetBlock,
    up_sample_layers=Up_B,
    seg_head_features=Seg_B,
)
FFNet_ResNet46N_CBB = partial(
    FFNet,
    stem_layers=Stem_C,
    backbone_layers=ResNet46N,
    backbone_block=ResNetBlock,
    up_sample_layers=Up_B,
    seg_head_features=Seg_B,
)

# FFNet-Mobile
FFNet_ResNet122NS_CCC = partial(
    FFNet,
    stem_layers=Stem_C,
    backbone_layers=ResNet122NS,
    backbone_block=ResNetBlock,
    up_sample_layers=Up_C,
    seg_head_features=Seg_C,
    mode="Mobile",
)
FFNet_ResNet74NS_CCC = partial(
    FFNet,
    stem_layers=Stem_C,
    backbone_layers=ResNet74NS,
    backbone_block=ResNetBlock,
    up_sample_layers=Up_C,
    seg_head_features=Seg_C,
    mode="Mobile",
)
FFNet_ResNet46NS_CCC = partial(
    FFNet,
    stem_layers=Stem_C,
    backbone_layers=ResNet46NS,
    backbone_block=ResNetBlock,
    up_sample_layers=Up_C,
    seg_head_features=Seg_C,
    mode="Mobile",
)

# DAB Net
DAB_Net = partial(DABNet)

# LEDNet
LEDNet = partial(LEDNet)
