#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from typing import Any, Sequence, Optional, Tuple
from .stochastic_depth import get_stochastic_depth_rate, StochasticDepth

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any

"""
    ConvNeXt v1 models for Flax.
    Reference:
        A ConvNet for the 2020s
        https://arxiv.org/abs/2201.03545
"""


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt backbone."""

    stage_sizes: Sequence[int]
    num_filters: Sequence[int]
    block_cls: ModuleDef
    conv: ModuleDef
    norm: ModuleDef
    stochastic_depth: ModuleDef
    act: ModuleDef
    kernel_size: Tuple[int, int]
    init_stochastic_depth_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, x):
        x = self.conv(
            96,
            kernel_size=(4, 4),
            strides=(4, 4),
            name="conv_init",
        )(x)

        x = self.norm(name="ln_init")(x)

        for i, block_size in enumerate(self.stage_sizes):
            stochastic_depth_drop_rate = get_stochastic_depth_rate(
                self.init_stochastic_depth_rate, i + 2, 5
            )

            if i > 0:
                # layer norm
                x = self.norm()(x)
                
                # downsampling
                x = self.conv(
                    self.num_filters[i],
                    kernel_size=(2, 2),
                    strides=(2, 2),
                )(x)

            # stage
            for j in range(block_size):
                x = self.block_cls(
                    self.num_filters[i],
                    strides=(1, 1),
                    conv=self.conv,
                    norm=self.norm,
                    act=self.act,
                    stochastic_depth=self.stochastic_depth,
                    stochastic_depth_drop_rate=stochastic_depth_drop_rate,
                    kernel_size=self.kernel_size,
                    name="BottleneckConvNeXtBlock_{:02}_{:02}".format(i + 1, j + 1)
                )(x)

        return x


class ConvNeXt(nn.Module):
    """ConvNeXt v1."""

    stage_sizes: Sequence[int]
    num_filters: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    init_stochastic_depth_rate: Optional[float] = 0.0
    kernel_size: Tuple[int, int] = (7, 7)
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.LayerNorm,
            epsilon=1e-6,
            dtype=self.dtype,
        )
        act = partial(
            nn.activation.gelu,
            approximate=False
        )
        stochastic_depth = partial(StochasticDepth, deterministic=not train)
        backbone = partial(
            ConvNeXtBackbone,
            conv=conv,
            norm=norm,
            act=act,
            kernel_size=self.kernel_size,
            stochastic_depth=stochastic_depth,
            init_stochastic_depth_rate=self.init_stochastic_depth_rate,
        )
        x = backbone(
            stage_sizes=self.stage_sizes,
            num_filters=self.num_filters,
            block_cls=self.block_cls,
        )(x)

        x = jnp.mean(x, axis=(1, 2))
        x = norm()(x)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)

        x = jnp.asarray(x, self.dtype)
        return x
