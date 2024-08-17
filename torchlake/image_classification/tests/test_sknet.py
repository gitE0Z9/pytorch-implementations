from math import ceil
import pytest
import torch

from ..models.resnet.network import ResBlock
from ..models.sknet.model import SkNet
from ..models.sknet.network import BottleNeck, SelectiveKernel2d


def test_sk2d_forward_shape():
    x = torch.randn(2, 32, 13, 13)
    layer = SelectiveKernel2d(32, 32)
    y = layer(x)

    assert y.shape == torch.Size((2, 32, 13, 13))


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


@pytest.mark.parametrize(
    "name,input_channel,base_number,output_channel,block,stride",
    [
        ["bottleneck_first", 64, 128, 256, BottleNeck, 2],
        ["bottleneck_middle", 256, 128, 256, BottleNeck, 1],
    ],
)
@pytest.mark.parametrize("pre_activation", [False, True])
def test_resblock_forward_shape(
    name: str,
    input_channel: int,
    base_number: int,
    output_channel: int,
    block: BottleNeck,
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


@pytest.mark.parametrize("num_layer", [26, 50, 101, 152])
@pytest.mark.parametrize("pre_activation", [False, True])
def test_sknet_forward_shape(num_layer: int, pre_activation: bool):
    x = torch.randn(2, 3, 224, 224)
    model = SkNet(
        output_size=5,
        num_layer=num_layer,
        pre_activation=pre_activation,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 5))
