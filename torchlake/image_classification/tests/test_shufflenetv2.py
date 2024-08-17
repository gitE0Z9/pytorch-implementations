from math import ceil
import pytest
import torch

from ..models.shufflenetv2.network import BottleNeck, ResBlock
from ..models.shufflenetv2.model import ShuffleNetV2


@pytest.mark.parametrize("input_channel,output_channel", [(48, 96), (96, 96)])
def test_bottleneck_forward_shape(input_channel: int, output_channel: int):
    x = torch.randn(2, input_channel, 224, 224)
    model = BottleNeck(input_channel, output_channel)
    y = model(x)

    assert y.shape == torch.Size((2, output_channel, 224, 224))


@pytest.mark.parametrize(
    "input_channel,output_channel,stride",
    [
        (24, 48, 2),
        (48, 48, 1),
        (24, 116, 2),
        (116, 116, 1),
        (24, 176, 2),
        (176, 176, 1),
        (24, 244, 2),
        (244, 244, 1),
    ],
)
@pytest.mark.parametrize("groups", [1, 2])
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


@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("scale_factor", [2, 1.5, 1, 0.5])
def test_shufflenetv2_forward_shape(groups: int, scale_factor: float):
    x = torch.randn(2, 3, 224, 224)
    model = ShuffleNetV2(
        output_size=5,
        groups=groups,
        scale_factor=scale_factor,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 5))
