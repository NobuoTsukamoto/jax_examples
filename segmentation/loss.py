#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from miou_metrics import _calc_semantic_segmentation_confusion

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
    """
    Reference:
        https://github.com/tensorflow/models/blob/c44482ab303f6a13b33048ca6058877c06a6a2d1/official/vision/losses/segmentation_losses.py#L36
    """
    if class_weights is None:
        class_weights = jnp.ones(num_classes, dtype=jnp.float32)

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
    """
    OHEM Loss

    Reference:
        https://arxiv.org/abs/1604.03540v1
    """
    if class_weights is None:
        class_weights = jnp.ones(num_classes, dtype=jnp.float32)

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
        threshold = jax.lax.cond(
            prob[threshold_index] > threshold,
            lambda _: prob[threshold_index],
            lambda _: threshold,
            None,
        )

        kept_mask = (prob < threshold).astype(jnp.int32)

        def apply_ohem_loss_true_fun(_):
            new_updated_labels = updated_labels * kept_mask
            new_valid_mask = valid_mask * kept_mask
            return new_updated_labels, new_valid_mask

        def apply_ohem_loss_false_fun(_):
            return updated_labels, valid_mask.astype(jnp.int32)

        is_apply_ohem_loss = (min_kept < num_valid) & (num_valid > 0)
        updated_labels, valid_mask = jax.lax.cond(
            is_apply_ohem_loss,
            apply_ohem_loss_true_fun,
            apply_ohem_loss_false_fun,
            None,
        )

    # make the invalid region as ignore
    updated_labels = updated_labels + (1 - valid_mask) * ignore_label

    updated_labels = updated_labels.reshape((batch, height, width))
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


def recall_cross_entroy_loss(
    logits, labels, num_classes, ignore_label=255, class_weights=None, epsilon=1e-5
):
    """
    Recall Loss

    Striking the Right Balance: Recall Loss for Semantic Segmentation

    Reference:
        https://arxiv.org/abs/2106.14917
        https://github.com/PotatoTian/recall-semseg/blob/main/ptsemseg/loss/loss.py
    """

    valid_mask = jnp.not_equal(labels, ignore_label)
    updated_labels = jnp.where(valid_mask, labels, -jnp.ones_like(labels))

    # batch, height, width, class -> batch, height, width, 1
    pred = jnp.argmax(logits, axis=-1, keepdims=True)

    # Calculate Confution matrix.
    cm = _calc_semantic_segmentation_confusion(
        pred, updated_labels, num_classes, ignore_label
    )
    true_positives = jnp.diag(cm)
    false_negatives = jnp.sum(cm, axis=1) - true_positives

    # Calculate ground truth counts.
    gt_counter = false_negatives + true_positives
    gt_counter = jnp.where(gt_counter > 0, gt_counter, 1)

    # Calculate false negative counts
    fn_counter = false_negatives
    fn_counter = jnp.where(fn_counter > 0, fn_counter, 1)

    if class_weights is not None:
        class_weights = 0.5 * (fn_counter / gt_counter) + 0.5 * class_weights
    else:
        class_weights = fn_counter / gt_counter

    valid_mask = jnp.any(valid_mask, axis=-1).astype(jnp.float32)

    one_hot_labels = jnp.squeeze(
        jax.nn.one_hot(updated_labels, num_classes=num_classes), axis=3
    )
    cross_entropy_loss = optax.softmax_cross_entropy(
        logits=logits, labels=one_hot_labels
    )
    weight_mask = jnp.einsum(
        "...y,y->...",
        one_hot_labels,
        class_weights,
    )
    cross_entropy_loss *= weight_mask
    num_valid_values = jnp.sum(valid_mask)

    cross_entropy_loss *= valid_mask
    return jnp.sum(cross_entropy_loss) / (num_valid_values + epsilon)
