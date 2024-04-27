import pytest
import torch

from ..models.resnet.network import BottleNeck, ConvBlock, ResBlock
from ..models.resnet.model import ResNet


@pytest.mark.parametrize(
    "name,input_channel,base_number,output_channel",
    [
        ["first", 64, 64, 256],
        ["middle", 256, 64, 256],
    ],
)
def test_bottleneck_forward_shape(
    name: str,
    input_channel: int,
    base_number: int,
    output_channel: int,
):
    x = torch.randn(2, input_channel, 13, 13)
    layer = BottleNeck(input_channel, base_number)
    y = layer(x)

    assert y.shape == torch.Size((2, output_channel, 13, 13))


@pytest.mark.parametrize(
    "name,input_channel,base_number",
    [
        ["first", 64, 128],
        ["middle", 128, 128],
    ],
)
def test_convblock_forward_shape(name: str, input_channel: int, base_number: int):
    x = torch.randn(2, input_channel, 13, 13)
    layer = ConvBlock(input_channel, base_number)
    y = layer(x)

    assert y.shape == torch.Size((2, base_number, 13, 13))


@pytest.mark.parametrize(
    "name,input_channel,base_number,output_channel,block",
    [
        ["convnet_first", 64, 128, 128, ConvBlock],
        ["convnet_middle", 128, 128, 128, ConvBlock],
        ["bottleneck_first", 64, 64, 256, BottleNeck],
        ["bottleneck_middle", 256, 64, 256, BottleNeck],
    ],
)
def test_resblock_forward_shape(
    name: str,
    input_channel: int,
    base_number: int,
    output_channel: int,
    block: ConvBlock | BottleNeck,
):
    x = torch.randn(2, input_channel, 13, 13)
    layer = ResBlock(input_channel, base_number, output_channel, block)
    y = layer(x)

    assert y.shape == torch.Size((2, output_channel, 13, 13))


@pytest.mark.parametrize(
    "name,num_layer",
    [
        ["18", 18],
        ["34", 34],
        ["50", 50],
        ["101", 101],
        ["152", 152],
    ],
)
def test_resnet_forward_shape(name: str, num_layer: int):
    x = torch.randn(2, 3, 224, 224)
    model = ResNet(output_size=5, num_layer=num_layer)
    y = model(x)

    assert y.shape == torch.Size((2, 5))


@pytest.mark.parametrize(
    "name,num_layer",
    [
        ["18", 18],
        ["34", 34],
        ["50", 50],
        ["101", 101],
        ["152", 152],
    ],
)
def test_resnet2_forward_shape(name: str, num_layer: int):
    x = torch.randn(2, 3, 224, 224)
    model = ResNet(output_size=5, num_layer=num_layer, pre_activation=True)
    y = model(x)

    assert y.shape == torch.Size((2, 5))
