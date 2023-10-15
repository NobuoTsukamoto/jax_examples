"""Tests for flax.examples.imagenet.models."""

import jax
from absl.testing import absltest, parameterized
from clu import parameter_overview
from jax import numpy as jnp

from dab_net import DABNet

jax.config.update("jax_disable_most_optimizations", True)


class DABNetTest(parameterized.TestCase):
    """Test cases for DABNet model definition."""

    def test_fast_scnn_model(self):
        """Tests DABNet model definition and output (variables)."""
        rngs = {
            "params": jax.random.PRNGKey(0),
            "dropout": jax.random.PRNGKey(1),
        }
        model_def = DABNet(num_classes=19, output_size=(512, 1024), dtype=jnp.float32)
        variables = model_def.init(rngs, jnp.ones((1, 512, 1024, 3), jnp.float32))

        self.assertLen(variables, 2)
        self.assertLen(variables["params"], 31)

        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
