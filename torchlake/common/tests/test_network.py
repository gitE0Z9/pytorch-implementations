import torch

from ..models import (
    Bam2d,
    Cbam2d,
    DepthwiseSeparableConv2d,
    ResBlock,
    SqueezeExcitation2d,
    SpatialTransform2d,
    ImageNormalization,
)
from ..network import ConvBnRelu


class TestSqueezeExcitation2d:
    def test_output_shape(self):
        x = torch.randn(8, 16, 7, 7)

        model = SqueezeExcitation2d(16, 16)

        y = model(x)

        assert y.shape == torch.Size((8, 16, 7, 7))


class TestCbam2d:
    def test_output_shape(self):
        x = torch.randn(8, 16, 7, 7)

        model = Cbam2d(16, 16)

        y = model(x)

        assert y.shape == torch.Size((8, 16, 7, 7))


class TestBam2d:
    def test_output_shape(self):
        x = torch.randn(8, 16, 112, 112)

        model = Bam2d(16, 16)

        y = model(x)

        assert y.shape == torch.Size((8, 16, 112, 112))


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

        assert y.shape == torch.Size((8, 32, 7, 7))


class TestResBlock:
    def test_output_shape(self):
        x = torch.randn(8, 16, 7, 7)

        model = ResBlock(16, 32, ConvBnRelu(16, 32, 3, padding=1, activation=None))

        y = model(x)

        assert y.shape == torch.Size((8, 32, 7, 7))


class TestSpatialTransform2d:
    def test_output_shape(self):
        x = torch.randn(8, 16, 224, 224)

        model = SpatialTransform2d(16, 32)

        y = model(x)

        for output in y:
            assert output.shape == torch.Size((8, 16, 224, 224))


class TestImageNormalization:
    def test_output_shape(self):
        x = torch.randn(8, 3, 224, 224)

        model = ImageNormalization()

        y = model(x)

        assert y.shape == torch.Size((8, 3, 224, 224))

    def test_reverse_output_shape(self):
        x = torch.randn(8, 3, 224, 224)

        model = ImageNormalization()

        y = model.forward(x, reverse=True)

        assert y.shape == torch.Size((8, 3, 224, 224))
