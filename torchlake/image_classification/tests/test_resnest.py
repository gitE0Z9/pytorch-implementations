from math import ceil

import pytest
import torch


from ..models.resnest.model import ResNeSt
from ..models.resnest.network import BottleNeck, SplitAttention2d

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 13


class TestNetwork:
    def test_split_attention_2d_forward_shape(self):
        x = torch.randn(BATCH_SIZE, 64, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        layer = SplitAttention2d(64, 128)
        y = layer(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, 128, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    @pytest.mark.parametrize(
        "name,input_channel,hidden_dim,expansion_factor,stride",
        [
            ["first", 64, 64, 4, 2],
            ["middle", 256, 64, 4, 1],
        ],
    )
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_bottleneck_forward_shape(
        self,
        name: str,
        input_channel: int,
        hidden_dim: int,
        expansion_factor: int,
        stride: int,
        pre_activation: bool,
    ):
        OUTPUT_SIZE = ceil(FEATURE_MAP_SIZE / stride)
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)

        # 64 -> 128 -> 128 -> 256
        model = BottleNeck(
            input_channel,
            hidden_dim,
            stride,
            expansion_factor,
            pre_activation,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, hidden_dim * expansion_factor, OUTPUT_SIZE, OUTPUT_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("key", [50, 101, 152])
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_resnest_forward_shape(self, key: int, pre_activation: bool):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = ResNeSt(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            key=key,
            pre_activation=pre_activation,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
