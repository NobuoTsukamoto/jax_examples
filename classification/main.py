#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Copyright (c) 2024 Nobuo Tsukamoto
    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import jax
import tensorflow as tf
from absl import app, flags, logging
from clu import platform
from ml_collections import config_flags

import train
import summary

flags.DEFINE_enum(
    name="task",
    default="summarize",
    enum_values=[
        "train",
        "summarize",
    ],
    help="Select task to perform.",
)
flags.DEFINE_string(name="workdir", default=None, help="Directory to store model data.")
config_flags.DEFINE_config_file(
    name="config",
    default=None,
    help_string="File path to the training hyperparameter configuration.",
    lock_config=True,
)

FLAGS = flags.FLAGS


def main(argv):
    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    if FLAGS.task == "train":

        if FLAGS.workdir is None:
            logging.error("If you specify train, please specify the workdir flag")
            raise ValueError("workdir flag is None.")

        else:
            # Add a note so that we can tell which task is which JAX host.
            # (Depending on the platform task 0 is not guaranteed to be host 0)
            platform.work_unit().set_task_status(
                f"process_index: {jax.process_index()}, "
                f"process_count: {jax.process_count()}"
            )
            platform.work_unit().create_artifact(
                platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
            )

            train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.task == "summarize":
        summary.summarize(FLAGS.config)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
