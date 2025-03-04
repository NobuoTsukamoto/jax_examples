#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2025 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from typing import Any, Optional

import jax.numpy as jnp
from optax import EmaState
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import utils


def ema_v2(
    decay: float, debias: bool = True, accumulator_dtype: Optional[Any] = None
) -> base.GradientTransformation:
    """Compute an exponential moving average of past updates.

    Refference:
    - https://optax.readthedocs.io/en/latest/api/transformations.html#optax.ema
    - https://www.tensorflow.org/api_docs/python/tfm/optimization/ExponentialMovingAverage

    """

    accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)
    average_decay = decay

    def init_fn(params):
        return EmaState(
            count=jnp.zeros([], jnp.int32),
            ema=otu.tree_zeros_like(params, dtype=accumulator_dtype),
        )

    def update_fn(updates, state, params=None):
        del params
        new_decay = jnp.minimum(
            average_decay, (1.0 + state.count) / (10.0 + state.count)
        )
        count_inc = numerics.safe_increment(state.count)
        updates = new_ema = otu.tree_update_moment(
            updates, state.ema, new_decay, order=1
        )
        if debias:
            updates = otu.tree_bias_correction(new_ema, new_decay, count_inc)
        state_ema = otu.tree_cast(new_ema, accumulator_dtype)
        return updates, EmaState(count=count_inc, ema=state_ema)

    return base.GradientTransformation(init_fn, update_fn)
