#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""
import numpy as np
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

""" Input Pipline

    Besed on:
        https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
"""

IMAGE_SIZE = (2048, 1024)
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

# fmt: off
LABEL_ID = np.asarray([0, 0, 0, 0, 0, 0,
                       0, 1, 2, 0, 0, 3,
                       4, 5, 0, 0, 0, 6,
                       0, 7, 8, 9, 10, 11,
                       12, 13, 14, 15, 16, 0,
                       0, 17, 18, 19, 0], dtype=np.int32)
# fmt: on


class Augment(tf.keras.layers.Layer):
    def __init__(self, image_size=IMAGE_SIZE, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def normalize_image(input_image, input_mask, dtype=tf.float32):
    input_image = tf.cast(input_image, dtype)
    input_image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=input_image.dtype)
    input_image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=input_image.dtype)
    input_image = tf.image.convert_image_dtype(input_image, dtype)

    input_mask = tf.cast(input_mask, dtype=tf.int32)
    input_mask = tf.where(input_mask >= 34, 34, input_mask)
    input_mask = tf.cast(tf.gather(LABEL_ID, input_mask), dtype=tf.uint8)
    # input_mask = tf.image.convert_image_dtype(input_mask, tf.uint8)

    # return {"image": input_image, "segmentation_mask": input_mask}
    return input_image, input_mask


def create_split(
    dataset_builder,
    batch_size,
    train,
    dtype=tf.float32,
    image_size=IMAGE_SIZE,
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

    def load_image(datapoint):
        input_image = datapoint["image_left"]
        input_mask = datapoint["segmentation_label"]

        input_image, input_mask = normalize_image(input_image, input_mask, dtype=dtype)

        return {"image": input_image, "segmentation_mask": input_mask}

    ds = dataset_builder.as_dataset(split=split)
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 48
    ds = ds.with_options(options)

    if cache:
        ds = ds.cache()

    if train:
        ds = ds.repeat()
        ds = ds.shuffle(16 * batch_size, seed=0)

    ds = ds.map(load_image,  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    if not train:
        ds = ds.repeat()

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds
