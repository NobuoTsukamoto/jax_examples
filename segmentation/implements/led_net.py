#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any

""" LEDNet models for Flax.
    Reference:
        LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation
        https://arxiv.org/abs/1905.02423

        https://github.com/xiaoyufenfei/LEDNet
"""


class DownSampling(nn.Module):
    in_features: int
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        # Downsampling Unit
        channel = x.shape[-1]
        x1 = self.conv(
            self.in_features - channel,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
        )(x)
        x2 = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        x = jnp.concatenate([x1, x2], axis=-1)

        x = self.norm()(x)
        return self.act(x)


class ChannelShffle(nn.Module):
    groups: int

    @nn.compact
    def __call__(self, x):
        batch, height, width, channels = x.shape
        channels_per_group = channels // self.groups

        x = jnp.reshape(x, (batch, height, width, self.groups, channels_per_group))
        x = x.transpose(0, 1, 2, 4, 3)
        x = x.reshape(batch, height, width, channels)
        return x


class SplitShuffleNonBottleneck(nn.Module):
    """Split-Shuffle-Non-Bottleneck Module"""

    in_features: int
    dilation: int = 1
    dropout_rate: float = 0.3
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu
    train: bool = True

    @nn.compact
    def __call__(self, x):
        channel_shffle = partial(ChannelShffle)

        x1, x2 = jnp.split(x, 2, axis=-1)

        x1 = self.conv(
            self.in_features // 2, kernel_size=(3, 1), strides=(1, 1), padding="SAME"
        )(x)
        x1 = self.act(x1)
        x1 = self.conv(
            self.in_features // 2, kernel_size=(1, 3), strides=(1, 1), padding="SAME"
        )(x1)
        x1 = self.norm()(x1)
        x1 = self.act(x1)
        x1 = self.conv(
            self.in_features // 2,
            kernel_size=(3, 1),
            kernel_dilation=(self.dilation, 1),
            strides=(1, 1),
            padding="SAME",
        )(x1)
        x1 = self.act(x1)
        x1 = self.conv(
            self.in_features // 2,
            kernel_size=(1, 3),
            kernel_dilation=(1, self.dilation),
            strides=(1, 1),
            padding="SAME",
        )(x1)
        x1 = self.norm()(x1)
        x1 = self.act(x1)
        x1 = nn.Dropout(rate=self.dropout_rate)(x1, deterministic=not self.train)

        x2 = self.conv(
            self.in_features // 2, kernel_size=(1, 3), strides=(1, 1), padding="SAME"
        )(x)
        x2 = self.act(x2)
        x2 = self.conv(
            self.in_features // 2, kernel_size=(3, 1), strides=(1, 1), padding="SAME"
        )(x2)
        x2 = self.norm()(x2)
        x2 = self.act(x2)
        x2 = self.conv(
            self.in_features // 2,
            kernel_size=(1, 3),
            kernel_dilation=(1, self.dilation),
            strides=(1, 1),
            padding="SAME",
        )(x2)
        x2 = self.act(x2)
        x2 = self.conv(
            self.in_features // 2,
            kernel_size=(3, 1),
            kernel_dilation=(self.dilation, 1),
            strides=(1, 1),
            padding="SAME",
        )(x2)
        x2 = self.norm()(x2)
        x2 = self.act(x2)
        x2 = nn.Dropout(rate=self.dropout_rate)(x2, deterministic=not self.train)

        x3 = jnp.concatenate([x1, x2], axis=-1)
        x = x3 + x
        x = self.act(x)

        x = channel_shffle(groups=2)(x)
        return x


class Encoder(nn.Module):
    """Encoder Module."""

    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu
    train: bool = True

    @nn.compact
    def __call__(self, x):
        downsampling = partial(
            DownSampling, conv=self.conv, norm=self.norm, act=self.act
        )
        ss_nbt = partial(
            SplitShuffleNonBottleneck,
            conv=self.conv,
            norm=self.norm,
            act=self.act,
            train=self.train,
        )

        # Downsampling Unit
        x = downsampling(in_features=32)(x)

        # 3 x SS-nbt Unit
        x = ss_nbt(in_features=32)(x)
        x = ss_nbt(in_features=32)(x)
        x = ss_nbt(in_features=32)(x)

        # Downsampling Unit
        x = downsampling(in_features=64)(x)

        # 2 x SS-nbt Unit
        x = ss_nbt(in_features=64)(x)
        x = ss_nbt(in_features=64)(x)

        # Downsampling Unit
        x = downsampling(in_features=128)(x)

        x = ss_nbt(in_features=128, dilation=1)(x)
        x = ss_nbt(in_features=128, dilation=2)(x)
        x = ss_nbt(in_features=128, dilation=5)(x)
        x = ss_nbt(in_features=128, dilation=9)(x)
        x = ss_nbt(in_features=128, dilation=2)(x)
        x = ss_nbt(in_features=128, dilation=5)(x)
        x = ss_nbt(in_features=128, dilation=9)(x)
        x = ss_nbt(in_features=128, dilation=17)(x)

        return self.act(x)


class APN(nn.Module):
    """APN Module."""

    out_features: int
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        batch, height, width, channels = x.shape

        # global pooling branch
        x1 = jnp.mean(x, axis=(1, 2), keepdims=True)
        x1 = self.conv(
            self.out_features, kernel_size=(1, 1), strides=(1, 1), padding="SAME"
        )(x1)
        x1 = self.norm()(x1)
        x1 = self.act(x1)
        x1 = jax.image.resize(
            x1, shape=(batch, height, width, self.out_features), method="bilinear"
        )

        # middle branch
        x2 = self.conv(
            self.out_features, kernel_size=(1, 1), strides=(1, 1), padding="SAME"
        )(x)

        # 128 x 64 x 128 -> 64 x 32 x 1
        x3 = self.conv(1, kernel_size=(7, 7), strides=(2, 2), padding="SAME")(x)
        x3 = self.norm()(x3)
        x3 = self.act(x3)
        print("x3", x3.shape)

        # 64 x 32 x 1 -> 32 x 16 x 1
        x4 = self.conv(1, kernel_size=(5, 5), strides=(2, 2), padding="SAME")(x3)
        x4 = self.norm()(x4)
        x4 = self.act(x4)
        print("x4", x4.shape)

        # 32 x 16 x 1 -> 16 x 8 x 1
        x5 = self.conv(1, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x4)
        x5 = self.norm()(x5)
        x5 = self.act(x5)
        print("x5", x5.shape)

        # 64 x 32 x 1
        x3 = self.conv(1, kernel_size=(7, 7), strides=(1, 1), padding="SAME")(x3)
        x3 = self.norm()(x3)
        x3 = self.act(x3)

        # 32 x 16 x 1
        x4 = self.conv(1, kernel_size=(5, 5), strides=(1, 1), padding="SAME")(x4)
        x4 = self.norm()(x4)
        x4 = self.act(x4)

        # 16 x 8 x 1 -> 32 x 16 x 1
        x5 = self.conv(1, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x5)
        x5 = self.norm()(x5)
        x5 = self.act(x5)
        x5 = jax.image.resize(
            x5, shape=(batch, height // 4, width // 4, x5.shape[-1]), method="bilinear"
        )

        # 32 x 16 x 1 -> 64 x 32 x 1
        x4 = x4 + x5
        x4 = jax.image.resize(
            x4, shape=(batch, height // 2, width // 2, x4.shape[-1]), method="bilinear"
        )

        # 64 x 32 x 1 -> 128 x 64 x 1
        x3 = x3 + x4
        x3 = jax.image.resize(
            x3, shape=(batch, height, width, x3.shape[-1]), method="bilinear"
        )

        x2 = jnp.multiply(x2, x3)
        x = x2 + x1

        return x


class Decoder(nn.Module):
    """Encoder Module."""

    num_classes: int
    output_size: tuple[int, int]
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        apn = partial(APN, conv=self.conv, norm=self.norm, act=self.act)

        output_shape = (
            x.shape[0],
            self.output_size[0],
            self.output_size[1],
            self.num_classes,
        )

        # APN Module
        x = apn(out_features=self.num_classes)(x)

        # Upsampling
        x = jax.image.resize(
            x,
            shape=output_shape,
            method="bilinear",
        )
        return x


class LEDNet(nn.Module):
    """LEDNet"""

    num_classes: int
    output_size: tuple[int, int]
    dtype: Any = jnp.float32
    conv: ModuleDef = nn.Conv
    act: Callable = nn.relu
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        encoder = partial(Encoder, conv=conv, norm=norm, act=self.act, train=train)
        decoder = partial(Decoder, conv=conv, norm=norm, act=self.act)

        x = encoder()(x)
        x = decoder(num_classes=self.num_classes, output_size=self.output_size)(x)

        x = jnp.asarray(x, self.dtype)
        return {"output": x}
