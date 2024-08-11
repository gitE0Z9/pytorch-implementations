from math import ceil
import pytest
import torch

from ..models.ghostnet.model import GhostNet
from ..models.ghostnet.network import GhostBottleNeck, GhostLayer, GhostModule


@pytest.mark.parametrize(
    "name,width_multiplier",
    [
        ["alpha=1", 1],
        ["alpha=0.5", 0.5],
    ],
)
def test_ghostnet_forward_shape(
    name: str,
    width_multiplier: float,
):
    x = torch.randn(2, 3, 224, 224)
    model = GhostNet(
        output_size=5,
        width_multiplier=width_multiplier,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 5))


@pytest.mark.parametrize(
    "name,s,d",
    [
        ["s=2, d=1", 2, 1],
        ["s=2, d=3", 2, 3],
    ],
)
def test_ghost_module_forward_shape(
    name: str,
    s: int,
    d: int,
):
    x = torch.randn(2, 64, 7, 7)
    model = GhostModule(64, 96, s, d)
    y = model(x)

    assert y.shape == torch.Size((2, 96, 7, 7))


@pytest.mark.parametrize(
    "name,stride,s,d,enable_se",
    [
        ["stride=1, s=2, d=1, enable_se = true", 1, 2, 1, True],
        ["stride=1, s=2, d=3, enable_se = true", 1, 2, 3, True],
        ["stride=1, s=2, d=1, enable_se = false", 1, 2, 1, False],
        ["stride=1, s=2, d=3, enable_se = false", 1, 2, 3, False],
        ["stride=2, s=2, d=1, enable_se = true", 2, 2, 1, True],
        ["stride=2, s=2, d=3, enable_se = true", 2, 2, 3, True],
        ["stride=2, s=2, d=1, enable_se = false", 2, 2, 1, False],
        ["stride=2, s=2, d=3, enable_se = false", 2, 2, 3, False],
    ],
)
def test_ghost_bottleneck_forward_shape(
    name: str,
    stride: int,
    s: int,
    d: int,
    enable_se: bool,
):
    INPUT_SHAPE = 7
    OUTPUT_SHAPE = ceil(7 / stride)

    x = torch.randn(2, 64, INPUT_SHAPE, INPUT_SHAPE)
    model = GhostBottleNeck(
        64,
        96,
        3,
        stride,
        s,
        d,
        expansion_size=128,
        enable_se=enable_se,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 96, OUTPUT_SHAPE, OUTPUT_SHAPE))


@pytest.mark.parametrize(
    "name,stride,s,d,enable_se",
    [
        ["stride=1, s=2, d=1, enable_se = true", 1, 2, 1, True],
        ["stride=1, s=2, d=3, enable_se = true", 1, 2, 3, True],
        ["stride=1, s=2, d=1, enable_se = false", 1, 2, 1, False],
        ["stride=1, s=2, d=3, enable_se = false", 1, 2, 3, False],
        ["stride=2, s=2, d=1, enable_se = true", 2, 2, 1, True],
        ["stride=2, s=2, d=3, enable_se = true", 2, 2, 3, True],
        ["stride=2, s=2, d=1, enable_se = false", 2, 2, 1, False],
        ["stride=2, s=2, d=3, enable_se = false", 2, 2, 3, False],
    ],
)
def test_ghost_layer_forward_shape(
    name: str,
    stride: int,
    s: int,
    d: int,
    enable_se: bool,
):
    INPUT_SHAPE = 7
    OUTPUT_SHAPE = ceil(7 / stride)

    x = torch.randn(2, 64, INPUT_SHAPE, INPUT_SHAPE)
    model = GhostLayer(
        64,
        96,
        3,
        stride,
        s,
        d,
        128,
        enable_se=enable_se,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 96, OUTPUT_SHAPE, OUTPUT_SHAPE))
