import pathlib
import tempfile

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import random
import tensorflow as tf
import tensorflow_datasets as tfds

# Local imports.
import models
import train
from configs import default as default_lib


jax.config.update("jax_disable_most_optimizations", True)


class TrainTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        # Make sure tf does not allocate gpu memory.
        tf.config.experimental.set_visible_devices([], "GPU")

    def test_create_model_fast_scnn(self):
        """Tests creating model."""
        model = train.create_model(
            model_cls=models.Fast_SCNN, num_classes=19, half_precision=False
        )
        rng = jax.random.PRNGKey(0)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {
            "params": params_rng,
            "dropout": dropout_rng,
        }
        params, batch_stats = train.initialized(rngs, (1024, 2048), model)
        variables = {"params": params, "batch_stats": batch_stats}
        x = random.normal(random.PRNGKey(1), (1, 1024, 2048, 3))
        y = model.apply(variables, x, train=False)
        self.assertEqual(y.shape, (1, 1024, 2048, 19))

    def test_create_model_lraspp(self):
        """Tests creating model."""
        model = train.create_model(
            model_cls=models.LRASPP_MobileNetV3_Large,
            num_classes=19,
            half_precision=False,
        )
        rng = jax.random.PRNGKey(0)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {
            "params": params_rng,
            "dropout": dropout_rng,
        }
        params, batch_stats = train.initialized(rngs, (1024, 2048), model)
        variables = {"params": params, "batch_stats": batch_stats}
        x = random.normal(random.PRNGKey(1), (1, 1024, 2048, 3))
        y = model.apply(variables, x, train=False)
        self.assertEqual(y.shape, (1, 1024, 2048, 19))

    @parameterized.product(model=("Fast_SCNN",))
    def test_train_and_evaluate_fast_scnn(self, model):
        """Tests training and evaluation loop using mocked data."""
        # Create a temporary directory where tensorboard metrics are written.
        workdir = tempfile.mkdtemp()

        # Go two directories up to the root of the flax directory.
        example_root_dir = pathlib.Path(__file__).parents[1]
        data_dir = str(example_root_dir) + "/.tfds/metadata"
        print(data_dir)

        # Define training configuration
        config = default_lib.get_config()
        config.model = model
        config.batch_size = 1
        config.num_epochs = 1
        config.num_train_steps = 1
        config.steps_per_eval = 1
        config.optimizer = "sgd"

        with tfds.testing.mock_data(
            num_examples=1,
            data_dir=data_dir,
        ):
            train.train_and_evaluate(workdir=workdir, config=config)

    @parameterized.product(model=("LRASPP_MobileNetV3_Large",))
    def test_train_and_evaluate_lraspp(self, model):
        """Tests training and evaluation loop using mocked data."""
        # Create a temporary directory where tensorboard metrics are written.
        workdir = tempfile.mkdtemp()

        # Go two directories up to the root of the flax directory.
        example_root_dir = pathlib.Path(__file__).parents[1]
        data_dir = str(example_root_dir) + "/.tfds/metadata"
        print(data_dir)

        # Define training configuration
        config = default_lib.get_config()
        config.model = model
        config.batch_size = 1
        config.num_epochs = 1
        config.num_train_steps = 1
        config.steps_per_eval = 1
        config.optimizer = "adam"

        with tfds.testing.mock_data(
            num_examples=1,
            data_dir=data_dir,
        ):
            train.train_and_evaluate(workdir=workdir, config=config)


if __name__ == "__main__":
    absltest.main()
