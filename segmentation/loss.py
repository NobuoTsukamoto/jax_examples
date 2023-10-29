#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import jax
import jax.numpy as jnp
import optax


def cross_entropy_loss(
    logits,
    labels,
    num_classes,
    ignore_label,
    class_weights=None,
    label_smoothing=0.0,
    epsilon=1e-5,
):
    class_weights = class_weights if class_weights else [1] * num_classes

    valid_mask = jnp.not_equal(labels, ignore_label)
    updated_labels = jnp.where(valid_mask, labels, -jnp.ones_like(labels))

    one_hot_labels = jnp.squeeze(
        jax.nn.one_hot(updated_labels, num_classes=num_classes), axis=3
    )
    smoothing_one_hot_labels = (
        one_hot_labels * (1 - label_smoothing) + label_smoothing / num_classes
    )
    cross_entropy_loss = optax.softmax_cross_entropy(
        logits=logits, labels=smoothing_one_hot_labels
    )

    valid_mask = jnp.any(valid_mask, axis=-1).astype(jnp.float32)

    weight_mask = jnp.einsum(
        "...y,y->...",
        one_hot_labels,
        class_weights,
    )
    cross_entropy_loss *= weight_mask
    num_valid_values = jnp.sum(valid_mask)

    cross_entropy_loss *= valid_mask
    return jnp.sum(cross_entropy_loss) / (num_valid_values + epsilon)


def ohem_cross_entropy_loss(
    logits,
    labels,
    num_classes,
    ignore_label,
    min_kept=256,
    threshold=0.7,
    class_weights=None,
    epsilon=1e-5,
):
    class_weights = class_weights if class_weights else [1] * num_classes

    batch, height, width, channel = logits.shape
    updated_labels = jnp.ravel(labels).astype(jnp.int32)

    valid_mask = jnp.not_equal(updated_labels, ignore_label)
    num_valid = jnp.sum(valid_mask)
    updated_labels = jnp.where(
        valid_mask, updated_labels, -jnp.ones_like(updated_labels)
    )

    prob = jax.nn.softmax(logits, axis=-1)
    # (bach, hight, width, channel) -> (channel, batch, height, widht)
    prob = prob.transpose((3, 0, 1, 2))
    prob = prob.reshape((channel, -1))

    if min_kept < num_valid and num_valid > 0:
        # let the value which ignored greater than 1
        prob = prob + (1 - valid_mask)

        # get the prob of relevant label
        one_hot_labels = jax.nn.one_hot(updated_labels, num_classes=num_classes)
        one_hot_labels = one_hot_labels.transpose((1, 0))
        prob = prob * one_hot_labels
        prob = jnp.sum(prob, axis=0)

        if min_kept > 0:
            index = jnp.argsort(prob)
            threshold_index = index[min(len(index), min_kept) - 1]
            threshold_index = threshold_index.astype(jnp.int32)
            if prob[threshold_index] > threshold:
                threshold = prob[threshold_index]

            kept_mask = (prob < threshold).astype(jnp.int32)
            updated_labels = updated_labels * kept_mask
            valid_mask = valid_mask * kept_mask

    # make the invalid region as ignore
    updated_labels = updated_labels + (1 - valid_mask) * ignore_label

    updated_labels = updated_labels.reshape((batch, height, width, 1))
    one_hot_labels = jnp.squeeze(
        jax.nn.one_hot(updated_labels, num_classes=num_classes), axis=3
    )
    valid_mask = valid_mask.reshape((batch, height, width, 1)).astype(jnp.float32)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)

    weight_mask = jnp.einsum(
        "...y,y->...",
        one_hot_labels,
        class_weights,
    )
    loss = loss * weight_mask
    loss = loss * valid_mask
    return jnp.mean(loss) / (jnp.mean(valid_mask) + epsilon)
