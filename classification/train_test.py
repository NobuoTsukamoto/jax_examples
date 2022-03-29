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

    def test_create_model(self):
        """Tests creating model."""
        model = train.create_model(
            model_cls=models.MobileNetV2_10, num_classes=10, half_precision=False
        )
        params, batch_stats = train.initialized(random.PRNGKey(0), 224, model)
        variables = {"params": params, "batch_stats": batch_stats}
        x = random.normal(random.PRNGKey(1), (8, 224, 224, 3))
        y = model.apply(variables, x, train=False)
        self.assertEqual(y.shape, (8, 10))

    @parameterized.product(model=("MobileNetV2_10",))
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
            policy=tfds.testing.MockPolicy.USE_FILES,
            data_dir="~/tensorflow_datasets",
        ):
            train.train_and_evaluate(workdir=workdir, config=config)


if __name__ == "__main__":
    absltest.main()
