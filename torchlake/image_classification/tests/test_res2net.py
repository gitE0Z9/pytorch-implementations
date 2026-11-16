from math import ceil

import pytest
import torch

from ..models.res2net.model import Res2Net
from ..models.res2net.network import BottleNeck, Res2NetLayer

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 13


class TestNetwork:
    @pytest.mark.parametrize("input_channel,split,groups", [[32, 2, 16], [128, 4, 32]])
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_res2net_layer_forward_shape(
        self,
        input_channel: int,
        split: int,
        groups: int,
        pre_activation: bool,
    ):
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        layer = Res2NetLayer(input_channel, split, groups, pre_activation)
        y = layer(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    @pytest.mark.parametrize(
        "name,input_channel,hidden_dim,expansion_factor,stride",
        [
            ["first", 64, 64, 4, 2],
            ["middle", 256, 64, 4, 1],
        ],
    )
    @pytest.mark.parametrize("pre_activation", [False, True])
    @pytest.mark.parametrize("split", [2, 4])
    @pytest.mark.parametrize("groups", [4, 8])
    def test_bottleneck_forward_shape(
        self,
        name: str,
        input_channel: int,
        hidden_dim: int,
        expansion_factor: int,
        stride: int,
        pre_activation: bool,
        split: int,
        groups: int,
    ):
        OUTPUT_SIZE = ceil(FEATURE_MAP_SIZE / stride)

        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        layer = BottleNeck(
            input_channel,
            hidden_dim,
            stride=stride,
            expansion_factor=expansion_factor,
            pre_activation=pre_activation,
            split=split,
            groups=groups,
        )
        y = layer(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, hidden_dim * expansion_factor, OUTPUT_SIZE, OUTPUT_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("key", [50, 101, 152])
    @pytest.mark.parametrize("pre_activation", [False, True])
    @pytest.mark.parametrize("split", [2, 4])
    @pytest.mark.parametrize("groups", [4, 8])
    def test_res2net_forward_shape(
        self,
        key: int,
        pre_activation: bool,
        split: int,
        groups: int,
    ):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = Res2Net(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            pre_activation=pre_activation,
            key=key,
            split=split,
            groups=groups,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
