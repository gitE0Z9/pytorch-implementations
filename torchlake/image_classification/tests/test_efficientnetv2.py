import pytest
import torch

from ..models.efficientnetv2.model import EfficientNetV2


class TestEfficientNetV2:
    @pytest.mark.parametrize("key", ["s", "m", "l"])
    def test_forward_shape(self, key: str):
        x = torch.randn(2, 3, 224, 224)
        model = EfficientNetV2(output_size=5, key=key)
        y = model(x)

        assert y.shape == torch.Size((2, 5))
