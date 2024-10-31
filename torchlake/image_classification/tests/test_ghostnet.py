from math import ceil
import pytest
import torch

from ..models.ghostnet.model import GhostNet
from ..models.ghostnet.network import GhostBottleNeck, GhostLayer, GhostModule


class TestGhostNet:
    @pytest.mark.parametrize("width_multiplier", [1, 0.5])
    def test_forward_shape(self, width_multiplier: float):
        x = torch.randn(2, 3, 224, 224)
        model = GhostNet(
            output_size=5,
            width_multiplier=width_multiplier,
        )
        y = model(x)

        assert y.shape == torch.Size((2, 5))


class TestGhostModule:
    @pytest.mark.parametrize("s", [2])
    @pytest.mark.parametrize("d", [1, 3])
    def test_forward_shape(self, s: int, d: int):
        x = torch.randn(2, 64, 7, 7)
        model = GhostModule(64, 96, s, d)
        y = model(x)

        assert y.shape == torch.Size((2, 96, 7, 7))


class TestGhostBottleNeck:
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("s", [2])
    @pytest.mark.parametrize("d", [1, 3])
    @pytest.mark.parametrize("enable_se", [True, False])
    def test_forward_shape(self, stride: int, s: int, d: int, enable_se: bool):
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


class TestGhostLayer:
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("s", [2])
    @pytest.mark.parametrize("d", [1, 3])
    @pytest.mark.parametrize("enable_se", [True, False])
    def test_forward_shape(
        self,
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
