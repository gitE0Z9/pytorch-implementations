import pytest
import torch

from ..models.mobilenetv2.model import MobileNetV2
from ..models.mobilenetv2.network import (
    LinearBottleneck,
    InvertedResidualBlock,
)


@pytest.mark.parametrize(
    "name,resolution_multiplier,width_multiplier",
    [
        ["rho=224,alpha=1", 224, 1],
        ["rho=224,alpha=0.75", 224, 0.75],
        ["rho=224,alpha=0.5", 224, 0.5],
        ["rho=224,alpha=0.25", 224, 0.25],
        ["rho=128,alpha=1", 128, 1],
        ["rho=128,alpha=0.75", 128, 0.75],
        ["rho=128,alpha=0.5", 128, 0.5],
        ["rho=128,alpha=0.25", 128, 0.25],
    ],
)
def test_mobilenetv2_forward_shape(
    name: str,
    resolution_multiplier: int,
    width_multiplier: float,
):
    x = torch.randn(2, 3, resolution_multiplier, resolution_multiplier)
    model = MobileNetV2(output_size=5, width_multiplier=width_multiplier)
    y = model(x)

    assert y.shape == torch.Size((2, 5))


def test_linear_bottleneck_forward_shape():
    x = torch.randn(2, 64, 7, 7)
    model = LinearBottleneck(64, 96, expansion_ratio=6)
    y = model(x)

    assert y.shape == torch.Size((2, 96, 7, 7))


def test_inverted_residual_block_forward_shape():
    x = torch.randn(2, 64, 7, 7)
    model = InvertedResidualBlock(64, 96, expansion_ratio=6)
    y = model(x)

    assert y.shape == torch.Size((2, 96, 7, 7))
