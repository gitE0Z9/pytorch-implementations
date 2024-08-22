from math import ceil
import pytest
import torch

from ..models.resnext.model import ResNeXt
from ..models.resnext.network import BottleNeck, ConvBlock


@pytest.mark.parametrize(
    "name,input_channel,base_number,stride",
    [
        ["first", 64, 64, 2],
        ["middle", 64, 64, 1],
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
        ["first", 64, 128, 256, 2],
        ["middle", 256, 128, 256, 1],
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
    # 64 -> 128 -> 128 -> 256
    layer = BottleNeck(input_channel, base_number, stride, pre_activation)
    y = layer(x)

    assert y.shape == torch.Size((2, output_channel, OUTPUT_SIZE, OUTPUT_SIZE))


@pytest.mark.parametrize("key", [18, 34, 50, 101, 152])
@pytest.mark.parametrize("pre_activation", [False, True])
def test_resnext_forward_shape(key: int, pre_activation: bool):
    x = torch.randn(2, 3, 224, 224)
    model = ResNeXt(
        output_size=5,
        key=key,
        pre_activation=pre_activation,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 5))
