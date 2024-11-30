from typing import Callable
import pytest
import torch

from ..models.efficientnet.model import (
    EfficientNet,
    efficient_b0,
    efficient_b1,
    efficient_b2,
    efficient_b3,
    efficient_b4,
    efficient_b5,
    efficient_b6,
    efficient_b7,
    efficient_b8,
    efficient_l2,
)


class TestEfficientNet:
    @pytest.mark.parametrize("width_multiplier", [1, 1.2])
    @pytest.mark.parametrize("depth_multiplier", [1, 1.2])
    def test_forward_shape(
        self,
        width_multiplier: float,
        depth_multiplier: float,
    ):
        x = torch.randn(2, 3, 224, 224)
        model = EfficientNet(
            output_size=5,
            width_multiplier=width_multiplier,
            depth_multiplier=depth_multiplier,
        )
        y = model(x)

        assert y.shape == torch.Size((2, 5))

    @pytest.mark.parametrize(
        "resolution,model_func",
        [
            (224, efficient_b0),
            (240, efficient_b1),
            (260, efficient_b2),
            (300, efficient_b3),
            (380, efficient_b4),
            (456, efficient_b5),
            (528, efficient_b6),
            (600, efficient_b7),
            (672, efficient_b8),
            (800, efficient_l2),
        ],
    )
    def test_variants_shape(self, resolution: int, model_func: Callable):
        x = torch.randn(2, 3, resolution, resolution)
        model = model_func(output_size=5)
        y = model(x)

        assert y.shape == torch.Size((2, 5))
