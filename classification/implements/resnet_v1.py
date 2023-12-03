#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from typing import Any, Callable, Sequence

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any

"""
    ResNet v1 models for Flax.
    Reference:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385
"""


class ResNetBackbone(nn.Module):
    """ResNet backbone."""

    stage_sizes: Sequence[int]
    num_filters: Sequence[int]
    block_cls: ModuleDef
    conv: ModuleDef
    norm: ModuleDef
    act: Callable

    @nn.compact
    def __call__(self, x):
        x = self.conv(
            64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)

        x = self.norm(name="bn_init")(x)
        x = self.act(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters[i],
                    strides=strides,
                    conv=self.conv,
                    norm=self.norm,
                    act=self.act,
                )(x)

        return x


class ResNet(nn.Module):
    """ResNetV1."""

    stage_sizes: Sequence[int]
    num_filters: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
            axis_name="batch",
        )
        backbone = partial(
            ResNetBackbone,
            conv=conv,
            norm=norm,
            act=nn.relu,
        )
        x = backbone(
            stage_sizes=self.stage_sizes,
            num_filters=self.num_filters,
            block_cls=self.block_cls,
        )(x)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x
