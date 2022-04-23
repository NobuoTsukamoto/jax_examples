"""Tests for flax.examples.imagenet.models."""

import jax
from absl.testing import absltest, parameterized
from clu import parameter_overview
from jax import numpy as jnp

from lite_raspp import LiteRASPP, MobileNetV3, Large, Small

jax.config.update("jax_disable_most_optimizations", True)


class DeepLabTest(parameterized.TestCase):
    """Test cases for LiteRASPP MobileNet V3 model definition."""

    def test_lraspp_mobilenet_v3_large_model(self):
        """Tests L-RASPP MobileNet V3 Large model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = LiteRASPP(
            backbone=MobileNetV3, layers=Large, num_classes=19, dtype=jnp.float32
        )
        variables = model_def.init(rng, jnp.ones((1, 1024, 2048, 3), jnp.float32))

        self.assertLen(variables, 2)
        # L-RASPP MobileNet V3 Large will create parameters for the following layers:
        #   Segmentation head = 1
        #   L-RASPP = 1
        self.assertLen(variables["params"], 2)

        print(parameter_overview.get_parameter_overview(variables))

    def test_lraspp_mobilenet_v3_small_model(self):
        """Tests L-RASPP MobileNet V3 Small model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = LiteRASPP(
            backbone=MobileNetV3, layers=Small, num_classes=19, dtype=jnp.float32
        )
        variables = model_def.init(rng, jnp.ones((1, 1024, 2048, 3), jnp.float32))

        self.assertLen(variables, 2)
        # L-RASPP MobileNet V3 Small will create parameters for the following layers:
        #   Segmentation head = 1
        #   L-RASPP = 1
        self.assertLen(variables["params"], 2)

        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
