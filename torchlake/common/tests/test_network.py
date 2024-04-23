import torch

from ..network import ConvBnRelu


class TestConvBnRelu:
    def test_output_shape(self):
        x = torch.randn(8, 16, 7, 7)

        model = ConvBnRelu(16, 32, 3)

        y = model(x)

        assert y.shape == torch.Size((8, 32, 5, 5))
