#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2025 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from typing import Any

import jax

from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state


class TrainStateWithBatchNorm(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale
    ema_decay: float = 0.0
    ema_params: Any = None

    def apply_ema(self):
        return jax.tree_util.tree_map(
            lambda ema, param: (ema * self.ema_decay + param * (1.0 - self.ema_decay)),
            self.ema_params,
            self.params,
        )


class TrainStateWithoutBatchNorm(train_state.TrainState):
    dynamic_scale: dynamic_scale_lib.DynamicScale
    ema_decay: float = 0.0
    ema_params: Any = None

    def apply_ema(self):
        return jax.tree_util.tree_map(
            lambda ema, param: (ema * self.ema_decay + param * (1.0 - self.ema_decay)),
            self.ema_params,
            self.params,
        )
