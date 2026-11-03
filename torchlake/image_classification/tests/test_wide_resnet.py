from math import ceil

import pytest
import torch

from ..models.wide_resnet import WideResNetWithBottleneck, WideResNet
from ..models.wide_resnet.network import DropoutConvBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
FEATURE_MAP_SIZE = 13
OUTPUT_SIZE = 5


class TestNetwork:
    @pytest.mark.parametrize(
        "name,input_channel,output_channel,stride",
        [
            ["first", 16, 16, 2],
            ["middle", 16, 32, 1],
        ],
    )
    @pytest.mark.parametrize("dropout_prob", [0.1, 0.3, 0.5, 0.7, 0.9])
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_dropout_convblock_forward_shape(
        self,
        name: str,
        input_channel: int,
        output_channel: int,
        stride: int,
        dropout_prob: float,
        pre_activation: bool,
    ):
        OUTPUT_SIZE = ceil(FEATURE_MAP_SIZE / stride)
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)

        model = DropoutConvBlock(
            input_channel,
            output_channel,
            stride=stride,
            pre_activation=pre_activation,
            dropout_prob=dropout_prob,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, OUTPUT_SIZE, OUTPUT_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("deepening_factor", [1, 2])
    @pytest.mark.parametrize("widening_factor", [1, 2])
    @pytest.mark.parametrize("dropout_prob", [0, 0.5])
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_wide_resnet_forward_shape(
        self,
        deepening_factor: int,
        widening_factor: int,
        dropout_prob: float,
        pre_activation: bool,
    ):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        model = WideResNet(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            deepening_factor=deepening_factor,
            widening_factor=widening_factor,
            pre_activation=pre_activation,
            dropout_prob=dropout_prob,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))

    @pytest.mark.parametrize("widening_factor", [1, 2])
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_wide_resnet_with_bottleneck_forward_shape(
        self,
        widening_factor: int,
        pre_activation: bool,
    ):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        model = WideResNetWithBottleneck(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            widening_factor=widening_factor,
            pre_activation=pre_activation,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
