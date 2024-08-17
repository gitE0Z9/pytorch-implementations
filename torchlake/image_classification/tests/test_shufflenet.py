from math import ceil
import pytest
import torch
from torch import nn
from torch.testing import assert_close

from ..models.shufflenet.network import ChannelShuffleLayer, BottleNeck, ResBlock
from ..models.shufflenet.model import ShuffleNet


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 8])
def test_channel_shuffle_layer_forward_shape(groups: int):
    x = torch.randn(2, 48, 224, 224)
    layer = ChannelShuffleLayer(groups=groups)
    official_layer = nn.ChannelShuffle(groups)
    y, official_y = layer(x), official_layer(x)

    assert_close(y, official_y)


@pytest.mark.parametrize("input_channel,output_channel", [(48, 96), (96, 96)])
@pytest.mark.parametrize("groups", [1, 2, 3, 4, 8])
def test_bottleneck_forward_shape(input_channel: int, output_channel: int, groups: int):
    x = torch.randn(2, input_channel, 224, 224)
    model = BottleNeck(input_channel, output_channel, groups=groups)
    y = model(x)

    assert y.shape == torch.Size((2, output_channel, 224, 224))


@pytest.mark.parametrize(
    "input_channel,output_channel,stride,groups",
    [
        (24, 144, 2, 1),
        (144, 144, 1, 1),
        (24, 200, 2, 2),
        (200, 200, 1, 2),
        (24, 240, 2, 3),
        (240, 240, 1, 3),
        (24, 272, 2, 4),
        (272, 272, 1, 4),
        (24, 384, 2, 8),
        (384, 384, 1, 8),
    ],
)
def test_resblock_forward_shape(
    input_channel: int,
    output_channel: int,
    stride: int,
    groups: int,
):
    INPUT_SIZE = 7
    OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

    x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
    model = ResBlock(
        input_channel,
        output_channel,
        stride=stride,
        groups=groups,
    )
    y = model(x)

    assert y.shape == torch.Size((2, output_channel, OUTPUT_SIZE, OUTPUT_SIZE))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("scale_factor", [2, 1.5, 1, 0.5, 0.25])
def test_shufflenet_forward_shape(groups: int, scale_factor: float):
    x = torch.randn(2, 3, 224, 224)
    model = ShuffleNet(
        output_size=5,
        groups=groups,
        scale_factor=scale_factor,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 5))
