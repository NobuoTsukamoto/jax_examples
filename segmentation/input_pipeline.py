#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2023 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
from random import Random
import numpy as np
import jax
import tensorflow as tf

""" Input Pipline

    Besed on:
        https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
"""

IMAGE_SIZE = (1024, 2048)
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

# fmt: off
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L52-L99
LABEL_ID = np.asarray([255, 255, 255, 255, 255, 255,
                       255,   0,   1, 255, 255,   2,
                         3,   4, 255, 255, 255,   5,
                       255,   6,   7,   8,   9,  10,
                        11,  12,  13,  14,  15, 255,
                       255,  16,  17,  18, 255], dtype=np.int32)
# fmt: on


class Augment(tf.keras.layers.Layer):
    def __init__(
        self,
        input_image_size=(1024, 2048),
        crop_size=(1024, 2048),
        output_image_size=(1024, 2048),
        min_resize_value=0.5,
        max_resize_value=2.0,
        ignore_label=255,
        seed=42,
        dtype=tf.float32,
    ):
        super().__init__()
        self.input_image_size = input_image_size
        self.crop_size = crop_size
        self.output_image_size = output_image_size
        self.min_resize_value = min_resize_value
        self.max_resize_value = max_resize_value
        self.ignore_label = ignore_label
        self.input_dtype = dtype
        self.random_resize_factor = Random(seed)
        self.inputs_random_flip = tf.keras.layers.RandomFlip(
            mode="horizontal", seed=seed
        )
        self.inputs_random_crop = tf.keras.layers.RandomCrop(
            height=crop_size[0], width=crop_size[1], seed=seed
        )
        self.labels_random_flip = tf.keras.layers.RandomFlip(
            mode="horizontal", seed=seed
        )
        self.labels_random_crop = tf.keras.layers.RandomCrop(
            height=crop_size[0], width=crop_size[1], seed=seed
        )
        self.inputs_random_contrast = tf.keras.layers.RandomContrast(
            factor=0.6, seed=seed
        )

    def call(self, inputs, labels):
        resize_factor = self.random_resize_factor.uniform(
            self.min_resize_value, self.max_resize_value
        )
        new_height = int(self.input_image_size[0] * resize_factor)
        new_width = int(self.input_image_size[1] * resize_factor)
        inputs = tf.image.resize(
            inputs, (new_height, new_width), method=tf.image.ResizeMethod.BILINEAR
        )
        pad_along_height = (
            self.input_image_size[0] - new_height
            if new_height < self.input_image_size[0]
            else 0
        )
        pad_along_width = (
            self.input_image_size[1] - new_width
            if new_width < self.input_image_size[1]
            else 0
        )
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        inputs = tf.pad(
            inputs,
            [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode="CONSTANT",
            constant_values=0.0,
        )
        inputs = self.inputs_random_crop(inputs)
        inputs = self.inputs_random_flip(inputs)
        inputs = self.inputs_random_contrast(inputs)

        labels = tf.image.resize(
            labels,
            (new_height, new_width),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        labels = tf.pad(
            labels,
            [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode="CONSTANT",
            constant_values=self.ignore_label,
        )
        labels = self.labels_random_crop(labels)
        labels = self.labels_random_flip(labels)
        labels = tf.image.resize(
            labels,
            (self.output_image_size[0], self.output_image_size[1]),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        inputs, labels = normalize_image(inputs, labels, dtype=self.dtype)

        return {"image": inputs, "label": labels}


def normalize_image(input_image, input_mask, dtype=tf.float32):
    input_image = tf.cast(input_image, dtype)
    input_image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=input_image.dtype)
    input_image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=input_image.dtype)

    input_mask = tf.cast(input_mask, dtype=tf.int32)
    input_mask = tf.where(input_mask >= 34, 34, input_mask)
    input_mask = tf.cast(
        tf.gather(LABEL_ID, tf.cast(input_mask, dtype=tf.int32)), dtype=tf.uint8
    )

    return input_image, input_mask


def create_split(
    dataset_builder,
    batch_size,
    train,
    dtype=tf.float32,
    input_image_size=IMAGE_SIZE,
    min_resize_value=0.5,
    max_resize_value=2.0,
    output_image_size=IMAGE_SIZE,
    cache=False,
    ignore_label=255,
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

    def load_image_train(datapoint):
        input_image = datapoint["image_left"]
        input_mask = datapoint["segmentation_label"]
        return input_image, input_mask

    def load_image_val(datapoint):
        input_image = datapoint["image_left"]
        input_mask = datapoint["segmentation_label"]

        input_image, input_mask = normalize_image(input_image, input_mask, dtype=dtype)
        input_image = tf.image.resize(
            input_image, input_image_size, method=tf.image.ResizeMethod.BILINEAR
        )
        input_image = tf.image.convert_image_dtype(input_image, dtype)
        input_mask = tf.image.resize(
            input_mask, output_image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        input_mask = tf.cast(input_mask, dtype=tf.int32)
        return {"image": input_image, "label": input_mask}

    if train:
        ds = dataset_builder.as_dataset(split=split).map(
            load_image_train, num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        ds = dataset_builder.as_dataset(split=split).map(
            load_image_val, num_parallel_calls=tf.data.AUTOTUNE
        )

    options = tf.data.Options()
    options.threading.private_threadpool_size = 8
    ds = ds.with_options(options)

    if cache:
        ds = ds.cache()

    if train:
        ds = ds.shuffle(2 * batch_size, seed=42)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.repeat()
        ds = ds.map(
            Augment(
                input_image_size=input_image_size,
                crop_size=input_image_size,
                output_image_size=output_image_size,
                min_resize_value=min_resize_value,
                max_resize_value=max_resize_value,
                ignore_label=ignore_label,
                seed=42,
            )
        )

    if not train:
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.repeat()

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds
