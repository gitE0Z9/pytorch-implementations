import pytest
import torch

from ..models.residual_attention.model import ResidualAttentionNetwork
from ..models.residual_attention.network import AttentionModule, MaskBranch

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 13


class TestNetwork:
    @pytest.mark.parametrize(
        "name,input_channel,output_channel,num_skip",
        [
            ["first-skip-2", 64, 256, 2],
            ["middle-skip-2", 256, 256, 2],
            ["first-skip-1", 64, 256, 1],
            ["middle-skip-1", 256, 256, 1],
            ["first-skip-0", 64, 256, 0],
            ["middle-skip-0", 256, 256, 0],
        ],
    )
    def test_mask_branch_forward_shape(
        self,
        name: str,
        input_channel: int,
        output_channel: int,
        num_skip: int,
    ):
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        layer = MaskBranch(input_channel, output_channel, num_skip=num_skip)
        y = layer(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    @pytest.mark.parametrize(
        "name,input_channel,output_channel",
        [
            ["first", 64, 256],
            ["middle", 256, 256],
        ],
    )
    def test_attention_module_forward_shape(
        self,
        name: str,
        input_channel: int,
        output_channel: int,
    ):
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        layer = AttentionModule(input_channel, output_channel)
        y = layer(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("key", [56, 92])
    def test_residual_attention_network_forward_shape(self, key: int):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = ResidualAttentionNetwork(INPUT_CHANNEL, OUTPUT_SIZE, key=key)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
