#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
import math

import numpy as np
import jax
import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_datasets as tfds

""" Input Pipline

    Besed on:
        https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
        https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/segmentation_input.py

"""

IMAGE_SIZE = (1024, 2048)
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

# fmt: off
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L52-L99
LABEL_ID = np.asarray([255, 255, 255, 255, 255, 255,
                       255,   0,   1, 255, 255,   2,
                         3,   4, 255, 255, 255,   5,
                       255,   6,   7,   8,   9,  10,
                        11,  12,  13,  14,  15, 255,
                       255,  16,  17,  18, 255], dtype=np.int32)
# fmt: on


def _normalize_image(image):
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    return image


def _prepare_image_and_label(datapoint, input_image_size):
    label = tf.io.decode_image(datapoint["segmentation_label"], channels=1)
    label = tf.reshape(label, (1, input_image_size[0], input_image_size[1]))

    image = tf.io.decode_image(datapoint["image_left"], 3, dtype=tf.uint8)
    image = tf.reshape(image, (input_image_size[0], input_image_size[1], 3))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tfm.vision.preprocess_ops.normalize_image(
        image,
        tfm.vision.preprocess_ops.MEAN_RGB,
        tfm.vision.preprocess_ops.STDDEV_RGB
    )

    label = tf.cast(label, dtype=tf.int32)
    label = tf.where(label >= 34, 34, label)
    label = tf.cast(tf.gather(LABEL_ID, tf.cast(label, dtype=tf.int32)), dtype=tf.uint8)
    label = tf.cast(label, tf.float32)

    return image, label


def _resize_and_crop_image(
    image,
    desired_size,
    padded_size,
    aug_scale_min=1.0,
    aug_scale_max=1.0,
    seed=42,
    method=tf.image.ResizeMethod.BILINEAR,
):
    image_size = tf.cast(tf.shape(image)[0:2], tf.float32)
    random_jittering = not math.isclose(aug_scale_min, 1.0) or not math.isclose(
        aug_scale_max, 1.0
    )

    if random_jittering:
        random_scale = tf.random.uniform([], aug_scale_min, aug_scale_max, seed=seed)
        scaled_size = tf.round(random_scale * tf.cast(desired_size, tf.float32))
    else:
        scaled_size = tf.cast(desired_size, tf.float32)

    scale = tf.minimum(scaled_size[0] / image_size[0], scaled_size[1] / image_size[1])
    scaled_size = tf.round(image_size * scale)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size

    # Selects non-zero random offset (x, y) if scaled image is larger than
    # desired_size.
    if random_jittering:
        max_offset = scaled_size - tf.cast(desired_size, tf.float32)
        max_offset = tf.where(
            tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset
        )
        offset = max_offset * tf.random.uniform(
            [
                2,
            ],
            0,
            1,
            seed=seed,
        )
        offset = tf.cast(offset, tf.int32)
    else:
        offset = tf.zeros((2,), tf.int32)

    scaled_image = tf.image.resize(image, tf.cast(scaled_size, tf.int32), method=method)

    if random_jittering:
        scaled_image = scaled_image[
            offset[0] : offset[0] + desired_size[0],
            offset[1] : offset[1] + desired_size[1],
            :,
        ]

    output_image = scaled_image
    if padded_size is not None:
        output_image = tf.image.pad_to_bounding_box(
            scaled_image, 0, 0, padded_size[0], padded_size[1]
        )

    image_info = tf.stack(
        [
            image_size,
            tf.cast(desired_size, dtype=tf.float32),
            image_scale,
            tf.cast(offset, tf.float32),
        ]
    )
    return output_image, image_info


def _resize_and_crop_masks(masks, image_scale, output_size, offset):
    mask_size = tf.cast(tf.shape(masks)[1:3], tf.float32)
    num_channels = tf.shape(masks)[3]
    # Pad masks to avoid empty mask annotations.
    masks = tf.concat(
        [
            tf.zeros([1, mask_size[0], mask_size[1], num_channels], dtype=masks.dtype),
            masks,
        ],
        axis=0,
    )
    scaled_size = tf.cast(image_scale * mask_size, tf.int32)
    scaled_masks = tf.image.resize(
        masks, scaled_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    offset = tf.cast(offset, tf.int32)
    scaled_masks = scaled_masks[
        :,
        offset[0] : offset[0] + output_size[0],
        offset[1] : offset[1] + output_size[1],
        :,
    ]

    output_masks = tf.image.pad_to_bounding_box(
        scaled_masks, 0, 0, output_size[0], output_size[1]
    )
    # Remove padding.
    output_masks = output_masks[1::]
    return output_masks


def parse_train_data(
    datapoint,
    aug_scale_min,
    aug_scale_max,
    ignore_label=255,
    crop_size=None,
    input_image_size=IMAGE_SIZE,
    output_image_size=None,
    dtype=tf.float32,
):
    image, label = _prepare_image_and_label(datapoint, input_image_size)

    if crop_size:
        crop_size = list(crop_size)
        label = tf.reshape(label, [input_image_size[0], input_image_size[1], 1])

        if output_image_size:
            image = tf.image.resize(image, output_image_size, method="bilinear")
            label = tf.image.resize(label, output_image_size, method="nearest")

        image_mask = tf.concat([image, label], axis=2)
        image_mask_crop = tf.image.random_crop(
            image_mask, crop_size + [tf.shape(image_mask)[-1]]
        )
        image = image_mask_crop[:, :, :-1]
        label = tf.reshape(image_mask_crop[:, :, -1], [1] + crop_size)

    # Flips image randomly during training.
    image, _, label = tfm.vision.preprocess_ops.random_horizontal_flip(
        image, masks=label
    )

    train_image_size = crop_size if crop_size else output_image_size
    # Resizes and crops image.
    image, image_info = tfm.vision.preprocess_ops.resize_and_crop_image(
        image, train_image_size, train_image_size, aug_scale_min, aug_scale_max
    )

    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]

    # Pad label and make sure the padded region assigned to the ignore label.
    # The label is first offset by +1 and then padded with 0.
    label += 1
    label = tf.expand_dims(label, axis=3)
    label = tfm.vision.preprocess_ops.resize_and_crop_masks(
        label, image_scale, train_image_size, offset
    )

    label -= 1
    label = tf.where(tf.equal(label, -1), ignore_label * tf.ones_like(label), label)
    label = tf.squeeze(label, axis=0)
    valid_mask = tf.not_equal(label, ignore_label)

    # Cast image as self._dtype
    image = tf.cast(image, dtype=dtype)
    label = tf.cast(label, tf.uint8)

    return image, label, valid_mask, image_info


def parse_eval_data(
    datapoint,
    ignore_label=255,
    input_image_size=IMAGE_SIZE,
    output_image_size=None,
    dtype=tf.float32,
):
    image, label = _prepare_image_and_label(datapoint, input_image_size)

    # The label is first offset by +1 and then padded with 0.
    label += 1
    label = tf.expand_dims(label, axis=3)

    # Resizes and crops image.
    image, image_info = tfm.vision.preprocess_ops.resize_and_crop_image(
        image, output_image_size, output_image_size
    )

    # Resizes eval masks to match input image sizes. In that case, mean IoU
    # is computed on output_size not the original size of the images.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    label = tfm.vision.preprocess_ops.resize_and_crop_masks(
        label, image_scale, output_image_size, offset
    )

    label -= 1
    label = tf.where(tf.equal(label, -1), ignore_label * tf.ones_like(label), label)
    label = tf.squeeze(label, axis=0)

    valid_mask = tf.not_equal(label, ignore_label)

    # Cast image as self._dtype
    image = tf.cast(image, dtype=dtype)
    label = tf.cast(label, tf.uint8)

    return image, label, valid_mask, image_info


def create_split(
    dataset_builder,
    batch_size,
    train,
    dtype=tf.float32,
    input_image_size=IMAGE_SIZE,
    output_image_size=None,
    crop_image_size=None,
    min_resize_value=0.5,
    max_resize_value=2.0,
    ignore_label=255,
    cache=False,
):
    """Creates a split from the ImageNet dataset using TensorFlow Datasets.
    Args:
        dataset_builder: TFDS dataset builder for ImageNet.
        batch_size: the batch size returned by the data pipeline.
        train: Whether to load the train or evaluation split.
        dtype: data type of the image.
        image_size: The target size of the images.
        cache: Whether to cache the dataset.
        ignore_label: ignore label.
    Returns:
        A `tf.data.Dataset`.
    """
    shuffle_buffer_size = 16 * batch_size
    prefetch = 10

    if train:
        train_examples = dataset_builder.info.splits["train"].num_examples
        split_size = train_examples // jax.process_count()
        start = jax.process_index() * split_size
        split = "train[{}:{}]".format(start, start + split_size)
    else:
        validate_examples = dataset_builder.info.splits["validation"].num_examples
        split_size = validate_examples // jax.process_count()
        start = jax.process_index() * split_size
        split = "validation[{}:{}]".format(start, start + split_size)

    def decode_example(example):
        if train:
            input_image, input_mask, _, _ = parse_train_data(
                example,
                min_resize_value,
                max_resize_value,
                ignore_label=ignore_label,
                input_image_size=input_image_size,
                output_image_size=output_image_size,
                crop_size=crop_image_size,
                dtype=dtype,
            )
        else:
            input_image, input_mask, _, _ = parse_eval_data(
                example,
                ignore_label=ignore_label,
                input_image_size=input_image_size,
                output_image_size=output_image_size,
                dtype=dtype,
            )

        return {"image": input_image, "label": input_mask}

    ds = dataset_builder.as_dataset(
        split=split,
        decoders={
            "image_left": tfds.decode.SkipDecoding(),
            "segmentation_label": tfds.decode.SkipDecoding(),
        },
    )
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    ds = ds.with_options(options)

    if cache:
        ds = ds.cache()

    if train:
        ds = ds.repeat()
        ds = ds.shuffle(shuffle_buffer_size, seed=42)

    ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    if not train:
        ds = ds.repeat()

    ds = ds.prefetch(prefetch)

    return ds
