#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2025 Nobuo Tsukamoto
This software is released under the MIT License.
See the LICENSE file in the project root for more information.
"""

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_models as tfm
import ml_collections
from absl import logging


""" Input Pipline

    Besed on:
        https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
        https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py
"""

IMAGE_SIZE = 224
CROP_FRACTION = 0.875


def distorted_bounding_box_crop(
    image_bytes,
    bbox,
    min_object_covered=0.1,
    aspect_ratio_range=(0.75, 1.33),
    area_range=(0.05, 1.0),
    max_attempts=100,
    scope=None,
):
    shape = tf.image.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True,
    )
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

    return image


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_center_crop(
    image_bytes, image_size, resize_method=tf.image.ResizeMethod.BICUBIC
):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    # crop_fraction = image_size / (image_size + crop_padding)
    crop_padding = round(image_size * (1 / CROP_FRACTION - 1))
    padded_center_crop_size = tf.cast(
        (
            (image_size / (image_size + crop_padding))
            * tf.cast(tf.minimum(image_height, image_width), tf.float32)
        ),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack(
        [offset_height, offset_width, padded_center_crop_size, padded_center_crop_size]
    )
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize(image, [image_size, image_size], method=resize_method)
    return image


def _decode_and_random_crop(
    image_bytes, image_size, resize_method=tf.image.ResizeMethod.BICUBIC
):
    """Make a random crop of image_size."""
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = distorted_bounding_box_crop(
        image_bytes,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3.0 / 4, 4.0 / 3.0),
        area_range=(0.08, 1.0),
        max_attempts=10,
    )
    original_shape = tf.image.extract_jpeg_shape(image_bytes)
    bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

    image = tf.cond(
        bad,
        lambda: _decode_and_center_crop(image_bytes, image_size),
        lambda: tf.image.resize(image, [image_size, image_size], method=resize_method),
    )

    return image


def preprocess_for_train_efficientnet(
    image_bytes,
    config: ml_collections.ConfigDict,
    augmenter=None,
    random_erasing=None,
    dtype=tf.float32,
):
    """EfficientNet-style preprocessing for training.

    Args:
        image_bytes: tf.Tensor, image in bytes (JPEG).
        config: ml_collections.ConfigDict containing 'image_size'.
        dtype: final dtype (e.g. tf.float32 or tf.bfloat16)

    Returns:
        Preprocessed image tensor.
    """

    # Decode and random crop (Inception-style)
    image = _decode_and_random_crop(image_bytes, config.image_size)

    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    image = tf.reshape(image, [config.image_size, config.image_size, 3])

    # Apply autoaug or randaug.
    if augmenter is not None:
        image = augmenter.distort(image)

    # Convert to [0.0, 1.0]
    image = tf.image.convert_image_dtype(image, dtype)

    return image


def preprocess_for_eval_efficientnet(
    image_bytes, config: ml_collections.ConfigDict, dtype=tf.float32
):
    image = _decode_and_center_crop(image_bytes, config.image_size)
    image = tf.reshape(image, [config.image_size, config.image_size, 3])
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
    logging.info("Input pipeline: %s", config.input_pipeline_type)

    augmenter = None
    random_erasing = None
    postprocess_fn = None
    if train:
        train_examples = dataset_builder.info.splits["train"].num_examples
        split_size = train_examples // jax.process_count()
        start = jax.process_index() * split_size
        split = "train[{}:{}]".format(start, start + split_size)

        num_classes = dataset_builder.info.features["label"].num_classes

        logging.info("Normalize: %s", config.normalize)
        logging.info("Data augument type: %s", config.aug_type)

        if config.aug_type == "autoaug":
            logging.info(
                "augmentation_name: %s, cutout_const: %d, autoaug_translate_const: %d",
                config.autoaug_augmentation_name,
                config.autoaug_cutout_const,
                config.autoaug_translate_const,
            )
            augmenter = tfm.vision.augment.AutoAugment(
                augmentation_name=config.autoaug_augmentation_name,
                cutout_const=config.autoaug_cutout_const,
                translate_const=config.autoaug_translate_const,
            )
        elif config.aug_type == "randaug":
            logging.info(
                "num_layers: %d, magnitude: %f, cutout_const: %f, translate_const: %f, "
                "magnitude_std: %f, prob_to_apply: %f, exclude_ops :%s",
                config.randaug_num_layers,
                config.randaug_magnitude,
                config.randaug_cutout_const,
                config.randaug_translate_const,
                config.randaug_magnitude_std,
                config.randaug_prob_to_apply,
                config.randaug_exclude_ops,
            )
            augmenter = tfm.vision.augment.RandAugment(
                num_layers=config.randaug_num_layers,
                magnitude=config.randaug_magnitude,
                cutout_const=config.randaug_cutout_const,
                translate_const=config.randaug_translate_const,
                magnitude_std=config.randaug_magnitude_std,
                prob_to_apply=config.randaug_prob_to_apply,
                exclude_ops=config.randaug_exclude_ops,
            )

        logging.info("Random erasing: %s", config.random_erasing)
        if config.random_erasing:
            logging.info(
                "probability: %f, min_area: %f, max_area: %f, min_aspect: %f, "
                "max_aspect: %f, min_count: %d, max_count: %d, trials: %d",
                config.random_erasing_probability,
                config.random_erasing_min_area,
                config.random_erasing_max_area,
                config.random_erasing_min_aspect,
                config.random_erasing_max_aspect,
                config.random_erasing_min_count,
                config.random_erasing_max_count,
                config.random_erasing_trials,
            )
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

        logging.info("Mixup and Cutmix: %s", config.mixup_and_cutmix)
        if config.mixup_and_cutmix:
            logging.info(
                "mixup_alpha: %f, cutmix_alpha: %f, prob: %f, switch_prob: %f, "
                "label_smoothing: %f, num_classes: %d",
                config.mixup_and_cutmix_mixup_alpha,
                config.mixup_and_cutmix_cutmix_alpha,
                config.mixup_and_cutmix_prob,
                config.mixup_and_cutmix_switch_prob,
                config.mixup_and_cutmix_label_smoothing,
                num_classes,
            )
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
            image = preprocess_for_train_efficientnet(
                example["image"],
                config,
                augmenter=augmenter,
                random_erasing=random_erasing,
                dtype=dtype,
            )
        else:
            image = preprocess_for_eval_efficientnet(example["image"], config, dtype)
        return {"image": image, "label": example["label"]}

    def postprocess(example):
        image, label = postprocess_fn.distort(example["image"], example["label"])
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
        ds = ds.shuffle(config.shuffle_buffer_size, seed=config.seed)

    ds = ds.map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    if train and postprocess_fn is not None:
        ds = ds.map(postprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if not train:
        ds = ds.repeat()

    ds = ds.prefetch(config.prefetch)

    return ds
