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
from .layer_scale import LayerScale

from flax import linen as nn
import jax.numpy as jnp
import numpy as np

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
    batch_norm: ModuleDef
    stochastic_depth: ModuleDef
    stochastic_depth_rate: Sequence[float]
    layer_scale: ModuleDef
    act: ModuleDef
    kernel_size: Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        x = self.conv(
            96,
            kernel_size=(4, 4),
            strides=(4, 4),
            name="conv_init",
        )(x)

        x = self.batch_norm(name="ln_init")(x)

        num_stage = 0
        for i, block_size in enumerate(self.stage_sizes):
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
                    stochastic_depth_drop_rate=self.stochastic_depth_rate[num_stage],
                    kernel_size=self.kernel_size,
                    layer_scale=self.layer_scale,
                    name="BottleneckConvNeXtBlock_{:02}_{:02}".format(i + 1, j + 1),
                )(x)
                num_stage += 1

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
        stochastic_depth_rate = [
            x.item()
            for x in np.linspace(
                0, self.init_stochastic_depth_rate, sum(self.stage_sizes)
            )
        ]
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        batch_norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
            axis_name="batch",
        )
        norm = partial(
            nn.LayerNorm,
            epsilon=1e-6,
            dtype=self.dtype,
            reduction_axes=-1,
        )
        act = partial(nn.activation.gelu, approximate=False)
        stochastic_depth = partial(StochasticDepth, deterministic=not train)
        layer_scale = partial(LayerScale, dtype=self.dtype)
        backbone = partial(
            ConvNeXtBackbone,
            conv=conv,
            norm=norm,
            act=act,
            kernel_size=self.kernel_size,
            stochastic_depth=stochastic_depth,
            stochastic_depth_rate=stochastic_depth_rate,
            layer_scale=layer_scale,
            batch_norm=batch_norm,
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
