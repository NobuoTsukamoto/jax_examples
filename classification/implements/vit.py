#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from typing import Any, Sequence, Optional, Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

ModuleDef = Any

"""
    ViT models for Flax.
    Reference:
     - https://gihyo.jp/book/2022/978-4-297-13058-9
"""


class MultiHeadSelfAttention(nn.Module):
    """ViT Multi-Head Self Attention."""

    embedded_dim: int = 384
    head: int = 3
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32

    def setup(self):
        self.head_dim = self.embedded_dim // self.head
        self.sqrt_dh = self.head_dim**0.5

    @nn.compact
    def __call__(self, x, train: bool = True):
        linear = partial(nn.Dense, use_bias=False, dtype=self.dtype)

        batch, num_patch, _ = x.shape

        # embedded
        # (B, N, D) ~> (B, N, D)
        q = linear(self.embedded_dim)(x)
        k = linear(self.embedded_dim)(x)
        v = linear(self.embedded_dim)(x)

        # (B, N, D) -> (B, N, h, D//h)
        q = jnp.reshape(q, (batch, num_patch, self.head, self.head_dim))
        k = jnp.reshape(k, (batch, num_patch, self.head, self.head_dim))
        v = jnp.reshape(v, (batch, num_patch, self.head, self.head_dim))

        # (B, N, h, D//h) -> (B, h, N, D//h)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # (B, h, N, D//h) -> (B, h, D//h, N)
        k_T = jnp.transpose(k, (0, 1, 3, 2))
        # (B, h, N, D//h) x (B, h, D//h, N) -> (B, h, N, N)
        dots = (q @ k_T) / self.sqrt_dh
        attn = nn.softmax(dots, axis=-1)
        attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=not train)

        # (B, h, N, N) x (B, h, N, D//h) -> (B, h, N, D//h)
        out = attn @ v
        # (B, h, N, D//h) -> (B, N, h, D//h)
        out = jnp.transpose(out, (0, 2, 1, 3))
        # (B, N, h, D//h) -> (B, N, D)
        out = jnp.reshape(out, (batch, num_patch, self.embedded_dim))

        # (B, N, D) -> (B, N, D)
        out = linear(self.embedded_dim)(out)
        out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=not train)

        return out


class VitInputLayer(nn.Module):
    """Vit Input layer."""

    in_channels: int = 3
    embedded_dim: int = 384
    num_patch_row: int = 2
    image_size: int = 32
    dtype: Any = jnp.float32

    def setup(self):
        self.num_patch = self.num_patch_row**2
        self.patch_size = int(self.image_size // self.num_patch_row)

        # class token
        self.class_token = self.param(
            "class token",
            jax.nn.initializers.normal(1.0),
            (1, 1, self.embedded_dim),
            self.dtype,
        )
        # positional embedding
        self.pos_emb = self.param(
            "positional embedding",
            jax.nn.initializers.normal(1.0),
            (1, self.num_patch + 1, self.embedded_dim),
            self.dtype,
        )

    @nn.compact
    def __call__(self, x):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)

        # (B, H, W, C) -> (B, H/P, W/P, D)
        # e.g. (2, 32, 32, 3) -> (2, 2, 2, 384)
        z_0 = conv(
            self.embedded_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
        )(x)

        # (B, H/P, W/P, D) -> (B, Np, D)
        # Np = H*W/P^2
        # eg. NP = 32*32/16^2 = 4
        z_0 = jax.lax.collapse(z_0, 1, -1)

        # (B, Np, D) -> (B, N, D)
        # N = (Np + 1)
        z_0 = jnp.concatenate(
            [jnp.repeat(self.class_token, repeats=x.shape[0], axis=0), z_0], axis=1
        )

        # (B, N, D) -> (B, N, D)
        z_0 = z_0 + self.pos_emb

        return z_0
