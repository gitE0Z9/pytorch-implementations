from typing import Sequence
import pytest
import torch

from torchlake.common.models.resnet_feature_extractor import ResNetFeatureExtractor

from ..models.dfn.model import DFN
from ..models.dfn.network import RefinementResidualBlock, ChannelAttentionBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 512
FEATURE_MAP_SIZE = 16
NUM_CLASS = 21


class TestNetwork:
    def test_refinement_residual_block_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM + 1, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = RefinementResidualBlock(HIDDEN_DIM + 1, HIDDEN_DIM)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    def test_channel_attention_block_forward_shape(self):
        shallow_x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM + 1, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )
        deep_x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM * 2, FEATURE_MAP_SIZE // 2, FEATURE_MAP_SIZE // 2)
        )

        model = ChannelAttentionBlock(HIDDEN_DIM + 1, HIDDEN_DIM * 2, HIDDEN_DIM)
        y = model(shallow_x, deep_x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM + 1, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize(
        "output_stride,target_layers",
        (
            (4, ("1_1", "2_1", "3_1", "4_1", "output")),
            (8, ("2_1", "3_1", "4_1", "output")),
        ),
    )
    @pytest.mark.parametrize("hidden_dim", (256, 512))
    def test_dfn_forward_shape(
        self,
        hidden_dim: int,
        output_stride: int,
        target_layers: Sequence[str],
    ):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = ResNetFeatureExtractor("resnet50", trainable=False)
        backbone.fix_target_layers(target_layers)
        model = DFN(
            backbone,
            NUM_CLASS,
            hidden_dim=hidden_dim,
            output_stride=output_stride,
        )

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
