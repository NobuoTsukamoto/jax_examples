"""Tests for flax.examples.imagenet.models."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp

from fast_scnn import FastSCNN


jax.config.update("jax_disable_most_optimizations", True)


class FastSCNNTest(parameterized.TestCase):
    """Test cases for Fast-SCNN model definition."""

    def test_fast_scnn_model(self):
        """Tests Fast-SCNN model definition and output (variables)."""
        rngs = {
            'params': jax.random.PRNGKey(0),
            'dropout': jax.random.PRNGKey(1),
        }
        model_def = FastSCNN(num_classes=19, dtype=jnp.float32)
        variables = model_def.init(rngs, jnp.ones((1, 2048, 1024, 3), jnp.float32))

        print(jax.tree_map(lambda x: x.shape, variables))
        print(sum(x.size for x in jax.tree_leaves(variables)))
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


if __name__ == "__main__":
    absltest.main()
