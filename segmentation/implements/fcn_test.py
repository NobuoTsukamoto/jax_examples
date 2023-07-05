"""Tests for flax.examples.imagenet.models."""

import jax
from absl.testing import absltest, parameterized
from clu import parameter_overview
from jax import numpy as jnp

from fcn import FCN, ResNet50V2_layer, ResNetV2Backbone

jax.config.update("jax_disable_most_optimizations", True)


class FCNTest(parameterized.TestCase):
    """Test cases for FCN model definition."""

    def test_fcn_model(self):
        """Tests FCN model definition and output (variables)."""
        rngs = {
            "params": jax.random.PRNGKey(0),
            "dropout": jax.random.PRNGKey(1),
        }
        model_def = FCN(
            backbone=ResNetV2Backbone,
            layers=ResNet50V2_layer,
            num_classes=19,
            dtype=jnp.float32,
        )
        variables = model_def.init(rngs, jnp.ones((1, 224, 224, 3), jnp.float32))

        self.assertLen(variables, 2)
        # Fast-SCNN model will create parameters for the following layers:
        #   Backbone= 1
        #   FCN Head = 1
        #   Conv = 1
        self.assertLen(variables["params"], 3)

        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
