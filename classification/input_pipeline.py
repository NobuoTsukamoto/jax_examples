#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2022 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_models as tfm
import ml_collections


""" Input Pipline

    Besed on:
        https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
"""

IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def preprocess_for_train(
    image_bytes,
    config: ml_collections.ConfigDict,
    augmenter=None,
    random_erasing=None,
    dtype=tf.float32,
):
    """Preprocesses the given image for training.
    Args:
        image_bytes: `Tensor` representing an image binary of arbitrary size.
        dtype: data type of the image.
        image_size: image size.
    Returns:
        A preprocessed image `Tensor`.
    """
    image = tf.io.decode_image(image_bytes, channels=3)
    image.set_shape([None, None, 3])

    # Crops image.
    cropped_image = tfm.vision.preprocess_ops.random_crop_image(
        image, area_range=config.crop_area_range
    )

    image = tf.cond(
        tf.reduce_all(tf.equal(tf.shape(cropped_image), tf.shape(image))),
        lambda: tfm.vision.preprocess_ops.center_crop_image(image),
        lambda: cropped_image,
    )

    if config.aug_rand_horizontal_flip:
        image = tf.image.random_flip_left_right(image)

    # Color jitter.
    if config.color_jitter > 0:
        image = tfm.vision.color_jitter(
            image, config.color_jitter, config.color_jitter, config.color_jitter
        )

    # Resizes image.
    image = tf.image.resize(
        image,
        [config.image_size, config.image_size],
        method=tf.image.ResizeMethod.BILINEAR,
    )
    image.set_shape([config.image_size, config.image_size, 3])

    # Apply autoaug or randaug
    if augmenter is not None:
        image = augmenter.distort(image)

    # Three augmentation
    if config.three_augment:
        image = tfm.vision.augment.AutoAugment(
            augmentation_name="deit3_three_augment",
            translate_const=20,
        ).distort(image)

    # Normalizes image with mean and std pixel values.
    image = tfm.vision.preprocess_ops.normalize_image(
        image,
        offset=tfm.vision.preprocess_ops.MEAN_RGB,
        scale=tfm.vision.preprocess_ops.STDDEV_RGB,
    )

    # Random erasing after the image has been normalized
    if random_erasing is not None:
        image = random_erasing.distort(image)

    # Convert image to self._dtype.
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def preprocess_for_eval(
    image_bytes, config: ml_collections.ConfigDict, dtype=tf.float32
):
    """Preprocesses the given image for evaluation.
    Args:
        image_bytes: `Tensor` representing an image binary of arbitrary size.
        dtype: data type of the image.
        image_size: image size.
    Returns:
        A preprocessed image `Tensor`.
    """
    image = tf.io.decode_image(image_bytes, channels=3)
    image.set_shape([None, None, 3])

    # Center crops.
    image = tfm.vision.preprocess_ops.center_crop_image(
        image, config.center_crop_fraction
    )
    image = tf.image.resize(
        image,
        [config.image_size, config.image_size],
        method=tf.image.ResizeMethod.BILINEAR,
    )
    image.set_shape([config.image_size, config.image_size, 3])

    # Normalizes image with mean and std pixel values.
    image = tfm.vision.preprocess_ops.normalize_image(
        image,
        offset=tfm.vision.preprocess_ops.MEAN_RGB,
        scale=tfm.vision.preprocess_ops.STDDEV_RGB,
    )

    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def create_split(
    dataset_builder,
    batch_size,
    dtype,
    train: bool,
    config: ml_collections.ConfigDict,
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
    augmenter = None
    random_erasing = None
    postprocess_fn = None
    if train:
        train_examples = dataset_builder.info.splits["train"].num_examples
        split_size = train_examples // jax.process_count()
        start = jax.process_index() * split_size
        split = "train[{}:{}]".format(start, start + split_size)

        num_classes = dataset_builder.info.features["label"].num_classes

        if config.aug_type == "autoaug":
            augmenter = tfm.vision.augment.AutoAugment(
                augmentation_name=config.autoaug_augmentation_name,
                cutout_const=config.autoaug_cutout_const,
                translate_const=config.autoaug_translate_const,
            )
        elif config.aug_type == "randaug":
            augmenter = tfm.vision.augment.RandAugment(
                num_layers=config.randaug_num_layers,
                magnitude=config.randaug_magnitude,
                cutout_const=config.randaug_cutout_const,
                translate_const=config.randaug_translate_const,
                prob_to_apply=config.randaug_prob_to_apply,
                exclude_ops=config.randaug_exclude_ops,
            )

        if config.random_erasing:
            random_erasing = tfm.vision.augment.RandomErasing(
                probability=config.random_erasing_probability,
                min_area=config.random_erasing_min_area,
                max_area=config.random_erasing_max_area,
                min_aspect=config.random_erasing_min_aspect,
                max_aspect=config.random_erasing_max_aspect,
                min_count=config.random_erasing_min_count,
                max_count=config.random_erasing_max_count,
                trials=config.random_erasing_trials,
            )

        if config.mixup_and_cutmix:
            postprocess_fn = tfm.vision.augment.MixupAndCutmix(
                mixup_alpha=config.mixup_and_cutmix_mixup_alpha,
                cutmix_alpha=config.mixup_and_cutmix_cutmix_alpha,
                prob=config.mixup_and_cutmix_prob,
                switch_prob=config.mixup_and_cutmix_switch_prob,
                label_smoothing=config.mixup_and_cutmix_label_smoothing,
                num_classes=num_classes,
            )

    else:
        validate_examples = dataset_builder.info.splits["validation"].num_examples
        split_size = validate_examples // jax.process_count()
        start = jax.process_index() * split_size
        split = "validation[{}:{}]".format(start, start + split_size)

    def decode_example(example):
        if train:
            image = preprocess_for_train(
                example["image"],
                config,
                augmenter=augmenter,
                random_erasing=random_erasing,
                dtype=dtype,
            )
        else:
            image = preprocess_for_eval(example["image"], config, dtype)
        return {"image": image, "label": example["label"]}

    def postprocess(example):
        if train:
            image, label = postprocess_fn.distort(example["image"], example["label"])
        else:
            image = example["image"]
            label = example["label"]

        return {"image": image, "label": label}

    ds = dataset_builder.as_dataset(
        split=split,
        decoders={
            "image": tfds.decode.SkipDecoding(),
        },
    )
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    ds = ds.with_options(options)

    if config.cache:
        ds = ds.cache()

    if train:
        ds = ds.repeat()
        ds = ds.shuffle(16 * batch_size, seed=42)

    ds = ds.map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    if postprocess_fn is not None:
        ds = ds.map(postprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if not train:
        ds = ds.repeat()

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds
