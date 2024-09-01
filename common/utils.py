#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import jax
import tensorflow as tf


def get_input_dtype(half_precision):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == "tpu":
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else:
        input_dtype = tf.float32

    return input_dtype
