import torch

from ..network import ConvBnRelu, DepthwiseSeparableConv2d


class TestConvBnRelu:
    def test_output_shape(self):
        x = torch.randn(8, 16, 7, 7)

        model = ConvBnRelu(16, 32, 3)

        y = model(x)

        assert y.shape == torch.Size((8, 32, 5, 5))


class TestDepthwiseSeparableConv2d:
    def test_output_shape(self):
        x = torch.randn(8, 16, 7, 7)

        model = DepthwiseSeparableConv2d(16, 32, 3)

        y = model(x)

        assert y.shape == torch.Size((8, 32, 5, 5))
