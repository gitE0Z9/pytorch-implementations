from math import ceil

import pytest
import torch

from ..models.resnet.model import ResNet
from ..models.resnet.network import BottleNeck, ConvBlock, ResBlock


@pytest.mark.parametrize(
    "name,input_channel,base_number,stride",
    [
        ["first", 64, 128, 2],
        ["middle", 128, 128, 1],
    ],
)
@pytest.mark.parametrize("pre_activation", [False, True])
def test_convblock_forward_shape(
    name: str,
    input_channel: int,
    base_number: int,
    stride: int,
    pre_activation: bool,
):
    INPUT_SIZE = 13
    OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

    x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
    layer = ConvBlock(input_channel, base_number, stride, pre_activation)
    y = layer(x)

    assert y.shape == torch.Size((2, base_number, OUTPUT_SIZE, OUTPUT_SIZE))


@pytest.mark.parametrize(
    "name,input_channel,base_number,output_channel,stride",
    [
        ["first", 64, 64, 256, 2],
        ["middle", 256, 64, 256, 1],
    ],
)
@pytest.mark.parametrize("pre_activation", [False, True])
def test_bottleneck_forward_shape(
    name: str,
    input_channel: int,
    base_number: int,
    output_channel: int,
    stride: int,
    pre_activation: bool,
):
    INPUT_SIZE = 13
    OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

    x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
    layer = BottleNeck(input_channel, base_number, stride, pre_activation)
    y = layer(x)

    assert y.shape == torch.Size((2, output_channel, OUTPUT_SIZE, OUTPUT_SIZE))


@pytest.mark.parametrize(
    "name,input_channel,base_number,output_channel,block,stride",
    [
        ["convnet_first", 64, 128, 128, ConvBlock, 2],
        ["convnet_middle", 128, 128, 128, ConvBlock, 1],
        ["bottleneck_first", 64, 64, 256, BottleNeck, 2],
        ["bottleneck_middle", 256, 64, 256, BottleNeck, 1],
    ],
)
@pytest.mark.parametrize("pre_activation", [False, True])
def test_resblock_forward_shape(
    name: str,
    input_channel: int,
    base_number: int,
    output_channel: int,
    block: ConvBlock | BottleNeck,
    stride: int,
    pre_activation: bool,
):
    INPUT_SIZE = 13
    OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

    x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
    layer = ResBlock(
        input_channel,
        base_number,
        output_channel,
        block,
        stride,
        pre_activation,
    )
    y = layer(x)

    assert y.shape == torch.Size((2, output_channel, OUTPUT_SIZE, OUTPUT_SIZE))


@pytest.mark.parametrize("num_layer", [18, 34, 50, 101, 152])
@pytest.mark.parametrize("pre_activation", [False, True])
@pytest.mark.parametrize("version", ["A", "B", "C", "D"])
def test_resnet_forward_shape(num_layer: int, pre_activation: bool, version: str):
    if num_layer in [18, 34] and version != "A":
        pytest.skip("version B, C, D only work with bottleneck layer.")

    x = torch.randn(2, 3, 224, 224)
    model = ResNet(
        output_size=5,
        num_layer=num_layer,
        pre_activation=pre_activation,
        version=version,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 5))
