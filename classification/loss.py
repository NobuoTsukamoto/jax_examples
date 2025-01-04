"""
    Copyright (c) 2025 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""


import jax.numpy as jnp
import optax
from flax.training import common_utils

def cross_entropy_loss(logits, labels, num_classes):
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


