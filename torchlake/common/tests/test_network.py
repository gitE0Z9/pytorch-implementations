from math import ceil

import pytest
import torch
from torch import nn
from torch.testing import assert_close
from torchvision.ops import Conv2dNormActivation

from ..models import (
    Bam2d,
    Cbam2d,
    ChannelShuffle,
    CoordinateAttention2d,
    DepthwiseSeparableConv2d,
    HighwayBlock,
    ResBlock,
    SqueezeExcitation2d,
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


class TestCoordinateAttention2d:
    def test_output_shape(self):
        x = torch.randn(8, 16, 112, 112)

        model = CoordinateAttention2d(16, 16)

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


@pytest.mark.parametrize(
    "name,stride",
    [
        ["stride=1", 1],
        ["stride=2", 2],
    ],
)
class TestResBlock:
    def test_output_shape(self, name: str, stride: int):
        INPUT_SHAPE = 7
        OUTPUT_SHAPE = ceil(7 / stride)
        x = torch.randn(8, 16, INPUT_SHAPE, INPUT_SHAPE)

        model = ResBlock(
            16,
            32,
            Conv2dNormActivation(
                16,
                32,
                3,
                stride,
                activation_layer=None,
            ),
            stride,
        )

        y = model(x)

        assert y.shape == torch.Size((8, 32, OUTPUT_SHAPE, OUTPUT_SHAPE))


class TestHighwayBlock:
    def test_output_shape(self):
        x = torch.randn(8, 32, 7, 7)

        model = HighwayBlock(
            Conv2dNormActivation(32, 32, 3),
            Conv2dNormActivation(32, 32, 3),
        )

        y = model(x)

        assert y.shape == torch.Size((8, 32, 7, 7))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 8])
def test_channel_shuffle_layer_forward_shape(groups: int):
    x = torch.randn(2, 48, 224, 224)
    layer = ChannelShuffle(groups=groups)
    official_layer = nn.ChannelShuffle(groups)
    y, official_y = layer(x), official_layer(x)

    assert_close(y, official_y)
