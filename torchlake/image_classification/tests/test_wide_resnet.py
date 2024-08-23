from math import ceil

import pytest
import torch

from ..models import BottleneckWideResNet, WideResNet
from ..models.wide_resnet.network import DropoutConvBlock, BottleNeck


@pytest.mark.parametrize(
    "name,input_channel,base_number,stride",
    [
        ["first", 16, 16, 2],
        ["middle", 16, 32, 1],
    ],
)
@pytest.mark.parametrize("dropout_prob", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("pre_activation", [False, True])
def test_dropout_convblock_forward_shape(
    name: str,
    input_channel: int,
    base_number: int,
    stride: int,
    dropout_prob: float,
    pre_activation: bool,
):
    INPUT_SIZE = 13
    OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

    x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
    layer = DropoutConvBlock(
        input_channel,
        base_number,
        stride,
        pre_activation,
        dropout_prob,
    )
    y = layer(x)

    assert y.shape == torch.Size((2, base_number, OUTPUT_SIZE, OUTPUT_SIZE))


@pytest.mark.parametrize(
    "name,input_channel,base_number,output_channel,stride",
    [
        ["first", 64, 64, 256, 2],
        ["middle", 256, 64, 256, 1],
    ],
)
@pytest.mark.parametrize("widening_factor", [1, 2])
@pytest.mark.parametrize("pre_activation", [False, True])
def test_bottleneck_forward_shape(
    name: str,
    input_channel: int,
    base_number: int,
    output_channel: int,
    stride: int,
    pre_activation: bool,
    widening_factor: int,
):
    INPUT_SIZE = 13
    OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

    x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
    layer = BottleNeck(
        input_channel,
        base_number,
        stride,
        pre_activation,
        widening_factor,
    )
    y = layer(x)

    assert y.shape == torch.Size((2, output_channel, OUTPUT_SIZE, OUTPUT_SIZE))


@pytest.mark.parametrize("deepening_factor", [1, 2])
@pytest.mark.parametrize("widening_factor", [1, 2])
@pytest.mark.parametrize("enable_dropout", [False, True])
@pytest.mark.parametrize("dropout_prob", [0.1, 0.9])
@pytest.mark.parametrize("pre_activation", [False, True])
def test_wide_resnet_forward_shape(
    deepening_factor: int,
    widening_factor: int,
    enable_dropout: bool,
    dropout_prob: float,
    pre_activation: bool,
):
    x = torch.randn(2, 3, 224, 224)
    model = WideResNet(
        output_size=5,
        pre_activation=pre_activation,
        deepening_factor=deepening_factor,
        widening_factor=widening_factor,
        enable_dropout=enable_dropout,
        dropout_prob=dropout_prob,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 5))


@pytest.mark.parametrize("widening_factor", [1, 2])
@pytest.mark.parametrize("pre_activation", [False, True])
def test_bottleneck_wide_resnet_forward_shape(
    widening_factor: int,
    pre_activation: bool,
):
    x = torch.randn(2, 3, 224, 224)
    model = BottleneckWideResNet(
        output_size=5,
        pre_activation=pre_activation,
        widening_factor=widening_factor,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 5))
