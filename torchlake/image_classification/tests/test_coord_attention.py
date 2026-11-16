from math import ceil

import pytest
import torch

from ..models.coord_attention.model import CoordAttentionResNet
from ..models.coord_attention.network import BottleNeck, CoordinateAttention2d

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
HIDDEN_DIM = 16
FEATURE_MAP_SIZE = 13


class TestNetwork:
    def test_coord_attention_2d_forward_shape(self):
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = CoordinateAttention2d(HIDDEN_DIM, 16)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
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
        layer = BottleNeck(
            input_channel,
            hidden_dim,
            stride=stride,
            expansion_factor=expansion_factor,
            pre_activation=pre_activation,
        )
        y = layer(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, hidden_dim * expansion_factor, OUTPUT_SIZE, OUTPUT_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("key", [50, 101, 152])
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_coord_attention_resnet_forward_shape(self, key: int, pre_activation: bool):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = CoordAttentionResNet(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            key=key,
            pre_activation=pre_activation,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
