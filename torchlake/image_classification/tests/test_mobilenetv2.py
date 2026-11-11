import pytest
import torch

from ..models.mobilenetv2.model import MobileNetV2
from ..models.mobilenetv2.network import (
    LinearBottleneck,
    InvertedResidualBlock,
)

BATCH_SIZE = 2
INPUT_CHANNEL = 3
OUTPUT_SIZE = 5
HIDDEN_DIM = 64
FEATURE_MAP_SIZE = 7


class TestNetwork:
    def test_linear_bottleneck_forward_shape(self):
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = LinearBottleneck(HIDDEN_DIM, HIDDEN_DIM + 2, expansion_ratio=6)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    def test_inverted_residual_block_forward_shape(self):
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = InvertedResidualBlock(HIDDEN_DIM, HIDDEN_DIM + 2, expansion_ratio=6)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
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
        self,
        name: str,
        resolution_multiplier: int,
        width_multiplier: float,
    ):
        x = torch.randn(
            BATCH_SIZE,
            INPUT_CHANNEL,
            resolution_multiplier,
            resolution_multiplier,
        )
        model = MobileNetV2(
            output_size=OUTPUT_SIZE,
            width_multiplier=width_multiplier,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
