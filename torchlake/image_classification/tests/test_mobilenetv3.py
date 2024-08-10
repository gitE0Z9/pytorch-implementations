import pytest
import torch

from ..models.mobilenetv3.model import MobileNetV3
from ..models.mobilenetv3.network import (
    DepthwiseSeparableConv2dV3,
    LinearBottleneckV3,
    InvertedResidualBlockV3,
)


@pytest.mark.parametrize(
    "name,size",
    [
        ["large", "large"],
        ["small", "small"],
    ],
)
def test_mobilenetv3_forward_shape(name: str, size: str):
    x = torch.randn(2, 3, 224, 224)
    model = MobileNetV3(output_size=5, size=size)
    y = model(x)

    assert y.shape == torch.Size((2, 5))


@pytest.mark.parametrize(
    "name,enable_relu,enable_se",
    [
        ["enable_relu", True, False],
        ["enable_relu", False, False],
        ["enable_se", False, False],
        ["enable_se", False, True],
    ],
)
def test_linear_bottleneck_v3_forward_shape(
    name: str,
    enable_relu: bool,
    enable_se: bool,
):
    x = torch.randn(2, 64, 7, 7)
    model = LinearBottleneckV3(
        64,
        96,
        3,
        expansion_size=24,
        enable_relu=enable_relu,
        enable_se=enable_se,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 96, 7, 7))


@pytest.mark.parametrize(
    "name,enable_relu,enable_se",
    [
        ["enable_relu", True, False],
        ["enable_relu", False, False],
        ["enable_se", False, False],
        ["enable_se", False, True],
    ],
)
def test_inverted_residual_block_v3_forward_shape(
    name: str,
    enable_relu: bool,
    enable_se: bool,
):
    x = torch.randn(2, 64, 7, 7)
    model = InvertedResidualBlockV3(
        64,
        96,
        3,
        expansion_size=24,
        enable_relu=enable_relu,
        enable_se=enable_se,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 96, 7, 7))


def test_depthwise_separable_conv2d_v3_forward_shape():
    x = torch.randn(2, 64, 7, 7)
    model = DepthwiseSeparableConv2dV3(64, 96)
    y = model(x)

    assert y.shape == torch.Size((2, 96, 7, 7))
