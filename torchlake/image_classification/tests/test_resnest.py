import pytest
import torch

from ..models.resnest.model import ResNeSt
from ..models.resnest.network import BottleNeck, SplitAttention2d
from ..models.resnet.network import ResBlock


def test_sa2d_forward_shape():
    x = torch.randn(2, 64, 13, 13)
    layer = SplitAttention2d(64, 128)
    y = layer(x)

    assert y.shape == torch.Size((2, 128, 13, 13))


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
    "name,input_channel,base_number,output_channel,block",
    [
        ["bottleneck_first", 64, 64, 256, BottleNeck],
        ["bottleneck_middle", 256, 64, 256, BottleNeck],
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
        ["50", 50],
        ["101", 101],
        ["152", 152],
    ],
)
def test_resnest_forward_shape(name: str, num_layer: int):
    x = torch.randn(2, 3, 224, 224)
    model = ResNeSt(output_size=5, num_layer=num_layer)
    y = model(x)

    assert y.shape == torch.Size((2, 5))
