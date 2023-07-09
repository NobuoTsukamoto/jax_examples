"""Tests for flax.examples.ffnet.models."""

import jax
from absl.testing import absltest, parameterized
from clu import parameter_overview
from jax import numpy as jnp

from ff_net import (
    FFNet,
    Stem_A,
    Stem_B,
    Stem_C,
    Up_A,
    Up_B,
    Up_C,
    Seg_A,
    Seg_B,
    Seg_C,
    ResNet150,
    ResNet150S,
    ResNet122N,
    ResNet122NS,
    ResNet101S,
    ResNet78S
)
from common_layer import ResNetBlock, BottleneckResNetBlock

jax.config.update("jax_disable_most_optimizations", True)


class FFNetTest(parameterized.TestCase):
    """Test cases for FFNet ResNet150 AAA GPU-Large model definition."""

    def test_ffnet_resnet122ns_cbb_mobile_model(self):
        """Tests FFNet ResNet78S BCC Moblie model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = FFNet(
            stem_layers=Stem_C,
            backbone_layers=ResNet122NS,
            backbone_block=ResNetBlock,
            up_sample_layers=Up_B,
            seg_head_features=Seg_B,
            mode="Mobile",
            num_classes=19,
            dtype=jnp.float32,
        )
        variables = model_def.init(rng, jnp.ones((1, 512, 1024, 3), jnp.float32))

        self.assertLen(variables, 2)
        # FFNet ResNet 78S B-C-C model will create parameters for the following layers:
        #   Stem = 1
        #   Backbone / Encoder and Up-Head / Decoder = 1
        #   Segmentation Head = 1
        self.assertLen(variables["params"], 3)

        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
