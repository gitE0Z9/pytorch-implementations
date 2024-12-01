from typing import Callable
import pytest
import torch

from ..models.efficientnet.model import (
    EfficientNet,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_b8,
    efficientnet_l2,
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
            (224, efficientnet_b0),
            (240, efficientnet_b1),
            (260, efficientnet_b2),
            (300, efficientnet_b3),
            (380, efficientnet_b4),
            (456, efficientnet_b5),
            (528, efficientnet_b6),
            (600, efficientnet_b7),
            # (672, efficientnet_b8),
            # (800, efficientnet_l2),
        ],
    )
    def test_variants_shape(self, resolution: int, model_func: Callable):
        x = torch.randn(2, 3, resolution, resolution)
        model = model_func(output_size=5)
        y = model(x)

        assert y.shape == torch.Size((2, 5))
