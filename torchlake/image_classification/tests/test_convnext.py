import pytest
import torch

from ..models.convnext.model import ConvNeXt
from ..models.convnext.network import BottleNeck


class TestNetwork:
    def test_forward_shape(self):
        x = torch.randn(2, 96, 14, 14)
        layer = BottleNeck(96)
        y = layer(x)

        assert y.shape == torch.Size((2, 96, 14, 14))


class TestModel:
    @pytest.mark.parametrize("size", ["tiny", "small", "base", "large"])
    def test_forward_shape(self, size: str):
        x = torch.randn(2, 3, 224, 224)
        model = ConvNeXt(output_size=5, size=size)
        y = model(x)

        assert y.shape == torch.Size((2, 5))
