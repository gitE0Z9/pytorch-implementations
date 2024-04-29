import pytest
import torch

from ..models.mobilenet.model import MobileNetV1, MobileNetV2, MobileNetV3
from ..models.mobilenet.network import (
    LinearBottleneck,
    InvertedResidualBlock,
    DepthwiseSeparableConv2dV3,
    LinearBottleneckV3,
    InvertedResidualBlockV3,
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
def test_mobilenetv1_forward_shape(
    name: str,
    resolution_multiplier: int,
    width_multiplier: float,
):
    x = torch.randn(2, 3, resolution_multiplier, resolution_multiplier)
    model = MobileNetV1(output_size=5, width_multiplier=width_multiplier)
    y = model(x)

    assert y.shape == torch.Size((2, 5))


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
