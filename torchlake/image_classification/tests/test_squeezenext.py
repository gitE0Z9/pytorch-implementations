from math import ceil
import pytest
import torch

from ..models.squeezenext.model import SqueezeNeXt
from ..models.squeezenext.network import BottleNeck


@pytest.mark.parametrize("width_multiplier", [1, 1.5, 2])
@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_squeezenet_forward_shape(width_multiplier: float, version: int):
    x = torch.randn(2, 3, 224, 224)
    model = SqueezeNeXt(
        output_size=5,
        width_multiplier=width_multiplier,
        version=version,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 5))


@pytest.mark.parametrize(
    "input_channel,output_channel,stride",
    [
        [64, 32, 2],
        [32, 32, 1],
    ],
)
def test_bottleneck_forward_shape(input_channel: int, output_channel: int, stride: int):
    INPUT_SIZE = 13
    OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

    x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
    model = BottleNeck(input_channel, output_channel, stride)
    y = model(x)

    assert y.shape == torch.Size((2, output_channel, OUTPUT_SIZE, OUTPUT_SIZE))
