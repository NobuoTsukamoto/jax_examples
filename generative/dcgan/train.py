#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from absl import app
from absl import flags
from flax import linen as nn
from flax.training import train_state
import jax.numpy as jnp
import jax
from jax import random
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds


""" DCGan

"""


class Generator(nn.Module):
    @nn.compact
    def __call__(self, x):