from math import ceil
import pytest
import torch

from ..models.resnext.model import ResNeXt
from ..models.resnext.network import BottleNeck, ConvBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 13


class TestNetwork:
    @pytest.mark.parametrize(
        "name,input_channel,output_channel,stride",
        [
            ["first", 64, 64, 2],
            ["middle", 64, 64, 1],
        ],
    )
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_convblock_forward_shape(
        self,
        name: str,
        input_channel: int,
        output_channel: int,
        stride: int,
        pre_activation: bool,
    ):
        OUTPUT_SIZE = ceil(FEATURE_MAP_SIZE / stride)
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)

        model = ConvBlock(
            input_channel,
            output_channel,
            stride=stride,
            pre_activation=pre_activation,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, OUTPUT_SIZE, OUTPUT_SIZE)
        )

    @pytest.mark.parametrize(
        "name,input_channel,hidden_dim,expansion_factor,stride",
        [
            ["first", 64, 128, 2, 2],
            ["middle", 256, 128, 2, 1],
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
    @pytest.mark.parametrize("key", [18, 34, 50, 101, 152])
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_resnext_forward_shape(self, key: int, pre_activation: bool):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        model = ResNeXt(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            key=key,
            pre_activation=pre_activation,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
