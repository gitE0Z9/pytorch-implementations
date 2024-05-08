import pytest
import torch

from ..models.densenet.model import DenseNet
from ..models.densenet.network import DenseBlock, TransitionBlock


class TestDenseNet:
    def test_denseblock_forward_shape(self):
        x = torch.randn(2, 16, 13, 13)
        layer = DenseBlock(16, 4)
        y = layer(x)

        assert y.shape == torch.Size((2, 16 + 4 * 32, 13, 13))

    def test_transitionblock_forward_shape(self):
        x = torch.randn(2, 16, 13, 13)
        layer = TransitionBlock(16, 8)
        y = layer(x)

        assert y.shape == torch.Size((2, 8, 6, 6))

    @pytest.mark.parametrize(
        "name,num_layer",
        [
            ["121", 121],
            ["169", 169],
            ["201", 201],
            ["264", 264],
        ],
    )
    def test_densenet_forward_shape(self, name: str, num_layer: int):
        x = torch.randn(2, 3, 224, 224)
        model = DenseNet(output_size=5, num_layer=num_layer)
        y = model(x)

        assert y.shape == torch.Size((2, 5))
