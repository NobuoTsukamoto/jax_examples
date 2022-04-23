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

    @parameterized.product(model=("",))
    def test_create_model(self):
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

    def test_train_and_evaluate(self, model):
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

        with tfds.testing.mock_data(
            num_examples=1,
            # policy=tfds.testing.MockPolicy.USE_FILES,
            data_dir=data_dir,
        ):
            train.train_and_evaluate(workdir=workdir, config=config)


if __name__ == "__main__":
    absltest.main()
