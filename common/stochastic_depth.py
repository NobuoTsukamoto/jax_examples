#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax, random

from flax import linen as nn
from flax.linen.module import merge_param


KeyArray = jax.Array


class StochasticDepth(nn.Module):
    """Create a stochastic depth layer.

        Note: When using :meth:`Module.apply() `, make
        sure to include an RNG seed named ``'stochastic_depth'``.
        StochasticDepth isn't necessary for variable initialization.

        Reference
            - Deep Networks with Stochastic Depth
            - https://github.com/tensorflow/models/blob/v2.14.2/official/vision/modeling/layers/nn_layers.py#L226-L261
            - https://pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html#StochasticDepth
            - https://flax.readthedocs.io/en/latest/_modules/flax/linen/stochastic.html#Dropout

    Attributes:
        stochastic_depth_drop_rate: the stochastic depth probability.
            (_not_ the keep rate!)
        deterministic: if false the inputs are scaled by ``1 / (1 - rate)`` and
            masked, whereas if true, no mask is applied and the inputs are returned as is.
        rng_collection: the rng collection name to use when requesting an rng key
    """

    stochastic_depth_drop_rate: float
    deterministic: Optional[bool] = None
    rng_collection: str = "stochastic_depth"

    @nn.compact
    def __call__(
        self,
        inputs,
        deterministic: Optional[bool] = None,
        rng: Optional[KeyArray] = None,
    ):
        """Applies a random stochastic depth mask to the input.

        Args:
            inputs: the inputs that should be randomly masked.
            deterministic: if false the inputs are scaled by ``1 / (1 - rate)``
                and masked, whereas if true, no mask is applied and the inputs
                are returnedas is.
            rng: an optional PRNGKey used as the random key, if not specified,
                one will be generated using ``make_rng`` with the
                ``rng_collection`` name.

        Returns:
            The masked inputs reweighted to preserve mean.
        """
        deterministic = merge_param("deterministic", self.deterministic, deterministic)

        if (self.stochastic_depth_drop_rate == 0.0) or deterministic:
            return inputs

        if rng is None:
            rng = self.make_rng(self.rng_collection)

        keep_prob = 1.0 - self.stochastic_depth_drop_rate
        batch_size = inputs.shape[0]

        broadcast_shape = list([batch_size] + [1] * int(inputs.ndim - 1))

        mask = random.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
        mask = jnp.broadcast_to(mask, inputs.shape)
        return lax.select(mask, inputs / keep_prob, jnp.zeros_like(inputs))


def get_stochastic_depth_rate(init_rate, i, n):
    """Get drop connect rate for the ith block.

    Args:
        init_rate: A `float` of initial drop rate.
        i: An `int` of order of the current block.
        n: An `int` total number of blocks.

    Returns:
        Drop rate of the ith block.
    """
    if init_rate < 0 or init_rate > 1:
        raise ValueError("Initial drop rate must be within 0 and 1.")
    return init_rate * float(i) / n
