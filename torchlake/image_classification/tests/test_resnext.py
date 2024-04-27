import pytest
import torch

from ..models.resnext.model import ResNeXt
from ..models.resnext.network import BottleNeck, ConvBlock


def test_bottleneck_first_forward_shape():
    x = torch.randn(2, 64, 13, 13)
    # 64 -> 128 -> 128 -> 256
    layer = BottleNeck(64, 128)
    y = layer(x)

    assert y.shape == torch.Size((2, 256, 13, 13))


def test_bottleneck_middle_forward_shape():
    x = torch.randn(2, 256, 13, 13)
    # 256 -> 128 -> 128 -> 256
    layer = BottleNeck(256, 128)
    y = layer(x)

    assert y.shape == torch.Size((2, 256, 13, 13))


def test_convblock_first_forward_shape():
    x = torch.randn(2, 64, 13, 13)
    # 64 -> 128 -> 64
    layer = ConvBlock(64, 64)
    y = layer(x)

    assert y.shape == torch.Size((2, 64, 13, 13))


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
def test_resnext_forward_shape(name: str, num_layer: int):
    x = torch.randn(2, 3, 224, 224)
    model = ResNeXt(output_size=5, num_layer=num_layer)
    y = model(x)

    assert y.shape == torch.Size((2, 5))
