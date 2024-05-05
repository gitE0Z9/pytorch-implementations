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
    "name,input_channel,base_number,output_channel",
    [
        ["first", 64, 128, 256],
        ["middle", 256, 128, 256],
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
    "name,input_channel,base_number,output_channel,block",
    [
        ["bottleneck_first", 64, 128, 256, BottleNeck],
        ["bottleneck_middle", 256, 128, 256, BottleNeck],
    ],
)
def test_resblock_forward_shape(
    name: str,
    input_channel: int,
    base_number: int,
    output_channel: int,
    block: BottleNeck,
):
    x = torch.randn(2, input_channel, 13, 13)
    layer = ResBlock(input_channel, base_number, output_channel, block)
    y = layer(x)

    assert y.shape == torch.Size((2, output_channel, 13, 13))


@pytest.mark.parametrize(
    "name,num_layer",
    [
        ["26", 26],
        ["50", 50],
        ["101", 101],
        ["152", 152],
    ],
)
def test_sknet_forward_shape(name: str, num_layer: int):
    x = torch.randn(2, 3, 224, 224)
    model = SkNet(output_size=5, num_layer=num_layer)
    y = model(x)

    assert y.shape == torch.Size((2, 5))
