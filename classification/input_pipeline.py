#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

from typing import List, Optional, Tuple, Any

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import ml_collections
import tensorflow_models as tfm

""" Input Pipline

    Besed on:
        https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
"""

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class ClassificationArgument:
    def __init__(
        self,
        config: ml_collections.ConfigDict,
        dtype: Optional[Any] = tf.float32,
    ):
        self._output_size = config.image_size
        self._aug_rand_horizontal_flip = config.aug_rand_horizontal_flip

        if config.aug_type == "autoaug":
            self._augmenter = tfm.vision.augment.AutoAugment(
                augmentation_name=config.autoaug_augmentation_name,
                cutout_const=config.autoaug_cutout_const,
                translate_const=config.autoaug_translate_const,
            )
        elif config.aug_type == "randaug":
            self._augmenter = tfm.vision.augment.RandAugment(
                num_layers=config.randaug_num_layers,
                magnitude=config.randaug_magnitude,
                cutout_const=config.randaug_cutout_const,
                translate_const=config.randaug_translate_const,
                prob_to_apply=config.randaug_prob_to_apply,
                exclude_ops=config.randaug_exclude_ops,
            )
        else:
            self._augmenter = None

        self._color_jitter = config.color_jitter

        if config.random_erasing:
            self._random_erasing = tfm.vision.augment.RandomErasing(
                probability=config.random_erasing.probability,
                min_area=config.random_erasing_min_area,
                max_area=config.random_erasing_max_area,
                min_aspect=config.random_erasing_min_aspect,
                max_aspect=config.random_erasing_max_aspect,
                min_count=config.random_erasing_min_count,
                max_count=config.random_erasing_max_count,
                trials=config.random_erasing_trials,
            )
        else:
            self._random_erasing = None

        self._crop_area_range = config.crop_area_range
        self._center_crop_fraction = config.center_crop_fraction
        self._three_augment = config.three_augment

        self._dtype = dtype

    def parse_train_image(self, image_bytes):
        image_shape = tf.image.extract_jpeg_shape(image_bytes)

        # Crops image.
        cropped_image = tfm.vision.preprocess_ops.random_crop_image_v2(
            image_bytes, image_shape, area_range=self._crop_area_range
        )
        image = tf.cond(
            tf.reduce_all(tf.equal(tf.shape(cropped_image), image_shape)),
            lambda: tfm.vision.preprocess_ops.center_crop_image_v2(
                image_bytes, image_shape
            ),
            lambda: cropped_image,
        )

        if self._aug_rand_horizontal_flip:
            image = tf.image.random_flip_left_right(image)

        # Color jitter.
        if self._color_jitter > 0:
            image = tfm.vision.color_jitter(
                image, self._color_jitter, self._color_jitter, self._color_jitter
            )

        # Resizes image.
        image = tf.image.resize(image, self._output_size, method="bilinear")
        image.set_shape([self._output_size[0], self._output_size[1], 3])

        # Apply autoaug or randaug
        if self._augmenter is not None:
            image = self._augmenter.distort(image)

        # Three augmentation
        if self._three_augment:
            image = tfm.vision.augment.AutoAugment(
                augmentation_name="deit3_three_augment",
                translate_const=20,
            ).distort(image)

        # Normalizes image with mean and std pixel values.
        image = tfm.vision.preprocess_ops.normalize_image(
            image, offset=MEAN_RGB, scale=STDDEV_RGB
        )

        # Random erasing after the image has been normalized
        if self._random_erasing is not None:
            image = self._random_erasing.distort(image)

        # Convert image to self._dtype.
        image = tf.image.convert_image_dtype(image, self._dtype)

        return image

    def parse_eval_image(self, image_bytes):
        image_shape = tf.image.extract_jpeg_shape(image_bytes)
        # Center crops.
        image = tfm.vision.preprocess_ops.center_crop_image_v2(
            image_bytes, image_shape, self._center_crop_fraction
        )
        image = tf.image.resize(image, self._output_size, method="bilinear")
        image.set_shape([self._output_size[0], self._output_size[1], 3])

        # Normalizes image with mean and std pixel values.
        image = tfm.vision.preprocess_ops.normalize_image(
            image, offset=MEAN_RGB, scale=STDDEV_RGB
        )

        # Convert image to self._dtype.
        image = tf.image.convert_image_dtype(image, self._dtype)

        return image


def create_split(
    dataset_builder,
    argument, 
    batch_size,
    train,
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

    def decode_example(example):
        if train:
            image = argument.parse_train_image(example["image"])
        else:
            image = argument.parse_eval_image(example["image"])
        return {"image": image, "label": example["label"]}

    ds = dataset_builder.as_dataset(
        split=split,
        decoders={
            "image": tfds.decode.SkipDecoding(),
        },
    )
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 48
    ds = ds.with_options(options)

    if cache:
        ds = ds.cache()

    if train:
        ds = ds.repeat()
        ds = ds.shuffle(16 * batch_size, seed=42)

    ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    if not train:
        ds = ds.repeat()

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds
