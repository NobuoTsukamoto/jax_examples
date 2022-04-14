"""Tests for flax.examples.imagenet.models."""

import jax
from absl.testing import absltest, parameterized
from clu import parameter_overview
from jax import numpy as jnp

from fast_scnn import FastSCNN

jax.config.update("jax_disable_most_optimizations", True)


class FastSCNNTest(parameterized.TestCase):
    """Test cases for Fast-SCNN model definition."""

    def test_fast_scnn_model(self):
        """Tests Fast-SCNN model definition and output (variables)."""
        rngs = {
            "params": jax.random.PRNGKey(0),
            "dropout": jax.random.PRNGKey(1),
        }
        model_def = FastSCNN(num_classes=19, dtype=jnp.float32)
        variables = model_def.init(rngs, jnp.ones((1, 1024, 2048, 3), jnp.float32))

        self.assertLen(variables, 2)
        # Fast-SCNN model will create parameters for the following layers:
        #   Conv + BatchNorm = 2
        #   DepthwiseSeparableConv = 2
        #   Bottleneck = 9
        #   PPM = 1
        #   FFM = 1
        #   DepthwiseSeparableConv = 2
        #   Conv = 1
        self.assertLen(variables["params"], 18)

        print(parameter_overview.get_parameter_overview(variables))

    def test_fast_scnn_1024_512_model(self):
        """Tests Fast-SCNN model definition and output (variables)."""
        rngs = {
            "params": jax.random.PRNGKey(0),
            "dropout": jax.random.PRNGKey(1),
        }
        model_def = FastSCNN(num_classes=19, dtype=jnp.float32)
        variables = model_def.init(rngs, jnp.ones((1, 1024, 512, 3), jnp.float32))

        self.assertLen(variables, 2)
        # Fast-SCNN model will create parameters for the following layers:
        #   Conv + BatchNorm = 2
        #   DepthwiseSeparableConv = 2
        #   Bottleneck = 9
        #   PPM = 1
        #   FFM = 1
        #   DepthwiseSeparableConv = 2
        #   Conv = 1
        self.assertLen(variables["params"], 18)
        
        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
