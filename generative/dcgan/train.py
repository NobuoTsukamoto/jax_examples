#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
from pyexpat import features
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from flax import linen as nn
from flax.training import train_state
from jax import random


""" DCGan

"""

FLAGS = flags.FLAGS

flags.DEFINE_float(
    "learning_rate", default=1e-3, help=("The learning rate for the Adam optimizer.")
)

flags.DEFINE_integer("batch_size", default=128, help=("Batch size for training."))

flags.DEFINE_integer("num_epochs", default=30, help=("Number of training epochs."))

flags.DEFINE_integer("nz", default=1000, help=("Size of z latent vector."))

flags.DEFINE_integer("generator_features", default=64, help=("generator features."))

flags.DEFINE_integer(
    "discriminator_features", default=64, help=("discriminator features.")
)


class Generator(nn.Module):
    dtype: Any = jnp.float32
    # Size of feature maps in generator
    generator_features: int = 64

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv_transpose = partial(nn.ConvTranspose, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        # input is Z, going into a convolution
        x = nn.Dense(features=4 * 4 * self.generator_features)(x)
        x = conv_transpose(self.generator_features * 8, (4, 4))(x)
        x = norm()(x)
        x = nn.leaky_relu(x)
        # state size. (generator_features x 8) x 4 x 4
        x = conv_transpose(self.generator_features * 4, (4, 4), strides=(2, 2))
        x = norm()(x)
        x = nn.leaky_relu(x)
        # state size. (generator_features x 4) x 8 x 8
        x = conv_transpose(self.generator_features * 2, (4, 4), strides=(2, 2))
        x = norm()(x)
        x = nn.leaky_relu(x)
        # state size. (generator_featuresngf x 2) x 16 x 16
        x = conv_transpose(self.generator_features, (4, 4), strides=(2, 2))
        x = norm()(x)
        x = nn.leaky_relu(x)
        # state size. 3 x 64 x 64
        x = conv_transpose(3, (4, 4), strides=(2, 2))
        x = jnp.tanh(x)

        return jnp.asarray(x, self.dtype)


class Discriminator(nn.Module):
    dtype: Any = jnp.float32
    # Size of feature maps in discriminator
    discriminator_features: int = 64
    dropout_rate: float = 4

    @nn.compact
    def __call__(self, x, train: bool = True, get_logits: bool = False):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )

        # input is n x 64 x 64 x 3
        x = conv(self.discriminator_features, (4, 4), strides=(2, 2))(x)
        x = norm(x)
        x = jnp.leaky_relu(x)

        # state size. (n x 32 x 32 x features
        x = conv(self.discriminator_features * 2, (4, 4), strides=(2, 2))(x)
        x = norm(x)
        x = jnp.leaky_relu(x)

        # state size. n x 16 x 16 x features * 2
        x = conv(self.discriminator_features * 4, (4, 4), strides=(2, 2))(x)
        x = norm(x)
        x = jnp.leaky_relu(x)

        # state size. n x 8 x 8 x features * 4
        x = conv(self.discriminator_features * 8, (4, 4), strides=(2, 2))(x)
        x = norm(x)
        x = jnp.leaky_relu(x)

        # state size. n x 4 x 4 x features * 8
        x = conv(1, (4, 4), strides=(1, 1), padding=(0, 0))(x)
        x = norm(x)
        x = jnp.leaky_relu(x)

        x = x.reshape([x.shape[0], -1])  # flatten
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(features=1)(x)

        if get_logits:
            return jnp.asarray(x, self.dtype)

        x = nn.sigmoid(x)
        return jnp.asarray(x, self.dtype)


class TrainState(train_state.TrainState):
    batch_stats: Any


def initialize_generator(key, nz, generator):
    input_shape = (1, nz, 1, 1)

    @jax.jit
    def init(*args):
        return generator.init(*args)

    variables = init({"params": key}, jnp.ones(input_shape, generator.dtype))
    return variables["params"], variables["batch_stats"]


def initialize_discriminator(key, discriminator):
    input_shape = (1, 64, 64, 3)

    @jax.jit
    def init(*args):
        return discriminator.init(*args)

    variables = init({"params": key}, jnp.ones(input_shape, discriminator.dtype))
    return variables["params"], variables["batch_stats"]


def create_generator(generator_features):
    model_dtype = jnp.float32

    return Generator(dtype=model_dtype, generator_features=generator_features)


def create_discriminator(discriminator_features):
    model_dtype = jnp.float32

    return Discriminator(
        dtype=model_dtype, discriminator_features=discriminator_features
    )


def create_train_state(rngs: Dict[str, jnp.ndarray], nz, generator):
    """Create initial training state."""

    generator_params, generator_batch_stats = initialize_generator(
        rngs[0], nz, generator
    )
    generator_tx = optax.adam(learning_rate=FLAGS.learning_rate)
    generator_state = TrainState.create(
        apply_fn=generator.apply,
        params=generator_params,
        tx=generator_tx,
        batch_stats=generator_batch_stats,
        sampling_rng=rngs[4],
    )
    return generator_state


def main(argv):
    del argv

    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], "GPU")

    ds_builder = tfds.builder("celeb_a")
    ds_builder.download_and_prepare()
    train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
    train_ds = train_ds.map(prepare_image)
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(50000)
    train_ds = train_ds.batch(FLAGS.batch_size)
    train_ds = iter(tfds.as_numpy(train_ds))

    rng = jax.random.PRNGKey(0)

    generator = create_generator(FLAGS.half_precision, FLAGS.generator_features)

    discriminator = create_discriminator(
        FLAGS.half_precision, FLAGS.discriminator_features
    )
