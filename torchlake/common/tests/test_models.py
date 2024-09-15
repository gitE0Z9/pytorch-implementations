from math import ceil, prod

import pytest
import torch
from torch import nn
from torch.testing import assert_close
from torchvision.ops import Conv2dNormActivation

from ..models import (
    ChannelShuffle,
    DepthwiseSeparableConv2d,
    FlattenFeature,
    HighwayBlock,
    KmaxPool1d,
    ResBlock,
    SqueezeExcitation2d,
    ImageNetNormalization,
    VggFeatureExtractor,
    ConvBnRelu,
)


class TestSqueezeExcitation2d:
    def test_output_shape(self):
        x = torch.randn(8, 16, 7, 7)

        model = SqueezeExcitation2d(16, 16)

        y = model(x)

        assert y.shape == torch.Size((8, 16, 7, 7))


class TestConvBnRelu:
    @pytest.mark.parametrize(
        "input_shape,output_shape,dimension",
        [
            [(8, 16, 7), (8, 32, 7), "1d"],
            [(8, 16, 7, 7), (8, 32, 7, 7), "2d"],
            [(8, 16, 7, 7, 7), (8, 32, 7, 7, 7), "3d"],
        ],
    )
    def test_output_shape(
        self,
        input_shape: tuple[int],
        output_shape: tuple[int],
        dimension: str,
    ):
        x = torch.randn(*input_shape)

        model = ConvBnRelu(16, 32, 3, padding=1, dimension=dimension)

        y = model(x)

        assert y.shape == torch.Size(output_shape)


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


@pytest.mark.parametrize(
    "input_shape,dimension",
    [[(7,), "1d"], [(7, 7), "2d"], [(7, 7, 7), "3d"]],
)
@pytest.mark.parametrize("start_dim", [1, 2])
@pytest.mark.parametrize("reduction", ["mean", "max", None])
def test_flatten_output_shape(
    input_shape: tuple[int],
    dimension: str,
    start_dim: int,
    reduction: str,
):
    x = torch.randn(8, 32, *input_shape)

    model = FlattenFeature(reduction, dimension, start_dim)

    y = model(x)

    reduced_factor = 1 if reduction is not None else prod(input_shape)
    if start_dim == 1:
        expected_shape = (8, 32 * reduced_factor)
    else:
        expected_shape = (8, 32, reduced_factor)
    assert y.shape == torch.Size(expected_shape)


def test_topk_pool_output_shape():
    x = torch.randn(8, 32, 7)

    model = KmaxPool1d(3)

    y = model(x)

    assert y.shape == torch.Size((8, 32, 3))
