#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from functools import partial
import jax.numpy as jnp
import jax

""" Jax mIoU metrics
    Based on chainer's eval_semantic_segmentation.

    chainercv.evaluations.eval_semantic_segmentation
"""


# https://github.com/google/jax/discussions/10078
@partial(jax.jit, static_argnames=("n", "ignore"))
def confusion_matrix(labels, predictions, n, ignore):
    cm, _ = jax.lax.scan(
        lambda carry, pair: (carry, None)
        if pair == ignore
        else (carry.at[pair].add(1), None),
        jnp.zeros((n, n), dtype=jnp.uint32),
        (labels, predictions),
    )
    return cm


def _calc_semantic_segmentation_confusion(
    pred_labels, gt_labels, num_classes, ignore_label
):
    batch_confusion_matrix = jax.vmap(confusion_matrix, in_axes=[0, 0, None, None])
    batch_cm = batch_confusion_matrix(gt_labels, pred_labels, num_classes, ignore_label)
    return batch_cm.sum(axis=0)


def _calc_semantic_segmentation_iou(confusion):
    iou_denominator = (
        confusion.sum(axis=1) + confusion.sum(axis=0) - jnp.diag(confusion)
    )
    iou = jnp.diag(confusion) / iou_denominator
    return iou


def eval_semantic_segmentation(pred_labels, gt_labels, num_classes, ignore_label=255):
    confusion = _calc_semantic_segmentation_confusion(
        pred_labels, gt_labels, num_classes, ignore_label
    )
    iou = _calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = jnp.diag(confusion).sum() / confusion.sum()
    class_accuracy = jnp.diag(confusion) / jnp.sum(confusion, axis=1)

    return {
        #       "iou": iou,
        "miou": jnp.nanmean(iou),
        "pixel_accuracy": pixel_accuracy,
        #       "class_accuracy": class_accuracy,
        "mean_class_accuracy": jnp.nanmean(class_accuracy),
    }
