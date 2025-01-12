#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2025 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from typing import Any

from flax import struct
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state

import optax


class TrainStateWithBatchNorm(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale
    ema_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    ema_state: optax.OptState = None


class TrainStateWithoutBatchNorm(train_state.TrainState):
    dynamic_scale: dynamic_scale_lib.DynamicScale
    ema_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    ema_state: optax.OptState = None
