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
    ResNet101S,
    ResNet78S
)
from common_layer import ResNetBlock, BottleneckResNetBlock

jax.config.update("jax_disable_most_optimizations", True)


class FFNetTest(parameterized.TestCase):
    """Test cases for FFNet ResNet150 AAA GPU-Large model definition."""

    def test_ffnet_resnet150_aaa_model(self):
        """Tests FCN model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = FFNet(
            stem_layers=Stem_A,
            backbone_layers=ResNet150,
            backbone_block=ResNetBlock,
            up_sample_layers=Up_A,
            seg_head_features=Seg_A,
            num_classes=19,
            dtype=jnp.float32,
        )
        variables = model_def.init(rng, jnp.ones((1, 1024, 2048, 3), jnp.float32))

        self.assertLen(variables, 2)
        # FFNet ResNet 150 A-A-A model will create parameters for the following layers:
        #   Stem = 1
        #   Backbone / Encoder and Up-Head / Decoder = 1
        #   Segmentation Head = 1
        self.assertLen(variables["params"], 3)

        print(parameter_overview.get_parameter_overview(variables))

    def test_ffnet_resnet150s_bbb_gpu_small_model(self):
        """Tests FFNet ResNet150S BBB GPU-Small model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = FFNet(
            stem_layers=Stem_B,
            backbone_layers=ResNet150S,
            backbone_block=ResNetBlock,
            up_sample_layers=Up_B,
            seg_head_features=Seg_B,
            mode="GPU-Small",
            num_classes=19,
            dtype=jnp.float32,
        )
        variables = model_def.init(rng, jnp.ones((1, 1024, 2048, 3), jnp.float32))

        self.assertLen(variables, 2)
        # FFNet ResNet 150S B-B-B model will create parameters for the following layers:
        #   Stem = 1
        #   Backbone / Encoder and Up-Head / Decoder = 1
        #   Segmentation Head = 1
        self.assertLen(variables["params"], 3)

        print(parameter_overview.get_parameter_overview(variables))

    def test_ffnet_resnet78s_bcc_mobile_model(self):
        """Tests FFNet ResNet78S BCC Moblie model definition and output (variables)."""
        rng = jax.random.PRNGKey(0)
        model_def = FFNet(
            stem_layers=Stem_B,
            backbone_layers=ResNet78S,
            backbone_block=ResNetBlock,
            up_sample_layers=Up_C,
            seg_head_features=Seg_C,
            mode="Mobile",
            num_classes=19,
            dtype=jnp.float32,
        )
        variables = model_def.init(rng, jnp.ones((1, 1024, 2048, 3), jnp.float32))

        self.assertLen(variables, 2)
        # FFNet ResNet 78S B-C-C model will create parameters for the following layers:
        #   Stem = 1
        #   Backbone / Encoder and Up-Head / Decoder = 1
        #   Segmentation Head = 1
        self.assertLen(variables["params"], 3)

        print(parameter_overview.get_parameter_overview(variables))


if __name__ == "__main__":
    absltest.main()
