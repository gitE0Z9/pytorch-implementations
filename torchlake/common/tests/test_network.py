import torch

from ..network import ConvBnRelu, DepthwiseSeparableConv2d, ResBlock


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


class TestResBlock:
    def test_output_shape(self):
        x = torch.randn(8, 16, 7, 7)

        model = ResBlock(16, 32, ConvBnRelu(16, 32, 3, padding=1, activation=None))

        y = model(x)

        assert y.shape == torch.Size((8, 32, 7, 7))