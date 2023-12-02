"""Tests for flax.examples.lednet.models."""

import jax
from absl.testing import absltest, parameterized
from clu import parameter_overview
from jax import numpy as jnp

from led_net import LEDNet

jax.config.update("jax_disable_most_optimizations", True)


class LEDNetTest(parameterized.TestCase):
    """Test cases for LEDNet model definition."""

    def test_led_net_model(self):
        """Tests LEDNet model definition and output (variables)."""
        rngs = {
            "params": jax.random.PRNGKey(0),
            "dropout": jax.random.PRNGKey(1),
            }
        model_def = LEDNet(
            num_classes=19,
            output_size=(512, 1024),
            dtype=jnp.float32,
        )
        variables = model_def.init(rngs, jnp.ones((1, 512, 1024, 3), jnp.float32))

        self.assertLen(variables, 2)
        # LEDNet model will create parameters for the following layers:
        #   encoder = 1
        #   decoder = 1
        self.assertLen(variables["params"], 2)

        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
