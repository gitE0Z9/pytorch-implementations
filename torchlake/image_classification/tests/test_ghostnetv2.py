from math import ceil

import pytest
import torch

from ..models.ghostnetv2.model import GhostNetV2
from ..models.ghostnetv2.network import DFCAttention, GhostBottleNeckV2, GhostLayerV2


class TestGhostNetV2:
    @pytest.mark.parametrize("width_multiplier", [1, 0.5])
    def test_forward_shape(self, width_multiplier: float):
        x = torch.randn(2, 3, 224, 224)
        model = GhostNetV2(
            output_size=5,
            width_multiplier=width_multiplier,
        )
        y = model(x)

        assert y.shape == torch.Size((2, 5))


class TestDFCAttention:
    @pytest.mark.parametrize(
        "horizontal_kernel,vertical_kernel", [(3, 3), (5, 5), (7, 7)]
    )
    def test_forward_shape(
        self,
        horizontal_kernel: int,
        vertical_kernel: int,
    ):
        x = torch.randn(2, 64, 14, 14)
        model = DFCAttention(
            64,
            96,
            horizontal_kernel,
            vertical_kernel,
        )
        y = model(x)

        assert y.shape == torch.Size((2, 96, 14, 14))


class TestGhostBottleNeckV2:
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("s", [2])
    @pytest.mark.parametrize("d", [1, 3])
    @pytest.mark.parametrize("enable_se", [True, False])
    def test_forward_shape(self, stride: int, s: int, d: int, enable_se: bool):
        INPUT_SHAPE = 7
        OUTPUT_SHAPE = ceil(7 / stride)

        x = torch.randn(2, 64, INPUT_SHAPE, INPUT_SHAPE)
        model = GhostBottleNeckV2(
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


class TestGhostLayerV2:
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("s", [2])
    @pytest.mark.parametrize("d", [1, 3])
    @pytest.mark.parametrize("enable_se", [True, False])
    def test_forward_shape(self, stride: int, s: int, d: int, enable_se: bool):
        INPUT_SHAPE = 7
        OUTPUT_SHAPE = ceil(7 / stride)

        x = torch.randn(2, 64, INPUT_SHAPE, INPUT_SHAPE)
        model = GhostLayerV2(
            64,
            96,
            3,
            stride=stride,
            s=s,
            d=d,
            expansion_size=128,
            enable_se=enable_se,
        )
        y = model(x)

        assert y.shape == torch.Size((2, 96, OUTPUT_SHAPE, OUTPUT_SHAPE))
