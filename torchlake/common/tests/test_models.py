from math import ceil, prod
from unittest import TestCase

import pytest
import torch
from torch import nn
from torch.testing import assert_close
from torchvision.ops import Conv2dNormActivation
from parameterized import parameterized
from ..models.kernel_pca import KernelEnum

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
    MultiKernelConvModule,
    KernelPCA,
    KMeans,
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


class TestHighwayBlock(TestCase):
    def test_output_shape(self):
        x = torch.randn(8, 32, 7, 7)

        model = HighwayBlock(
            Conv2dNormActivation(32, 32, 3),
            Conv2dNormActivation(32, 32, 3),
        )

        y = model(x)

        assert y.shape == torch.Size((8, 32, 7, 7))


class TestChannelShuffle:
    @pytest.mark.parametrize("groups", [1, 2, 3, 4, 8])
    def test_output_shape(self, groups: int):
        x = torch.randn(2, 48, 224, 224)
        layer = ChannelShuffle(groups=groups)
        official_layer = nn.ChannelShuffle(groups)
        y, official_y = layer(x), official_layer(x)

        assert_close(y, official_y)


class TestFlattenFeature:
    @pytest.mark.parametrize(
        "input_shape,dimension",
        [[(7,), "1d"], [(7, 7), "2d"], [(7, 7, 7), "3d"]],
    )
    @pytest.mark.parametrize("start_dim", [1, 2])
    @pytest.mark.parametrize("reduction", ["mean", "max", None])
    def test_output_shape(
        self,
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


class TestTopkPool(TestCase):
    def test_max_1d_output_shape(self):
        x = torch.randn(8, 32, 7)

        model = KmaxPool1d(3)

        y = model(x)

        assert y.shape == torch.Size((8, 32, 3))


class TestKernelPCA(TestCase):
    @parameterized.expand(
        [
            (
                "linear_kernel",
                KernelEnum.LINEAR,
            ),
            (
                "rbf_kernel",
                KernelEnum.RBF,
            ),
            (
                "helligner_kernel",
                KernelEnum.HELLINGER,
            ),
        ]
    )
    def test_output_shape(self, name: str, kernel: str):
        x = torch.randn(8, 10)
        kernel_params = {}

        if kernel == KernelEnum.HELLINGER:
            x = torch.randint(0, 100, (8, 10)).float()
            kernel_params["is_normalized"] = False

        model = KernelPCA(2, kernel, kernel_params)

        model.fit(x)

        assert model.eigenvalues.shape == torch.Size((2,))
        assert model.eigenvectors.shape == torch.Size((8, 2))


class TestKMeans:
    @pytest.mark.parametrize("k", [3, 5, 10])
    @pytest.mark.parametrize("init_method", ["random", "kmeans++"])
    def test_output_shape(self, k: int, init_method: str):
        x = torch.randn(100, 10)
        model = KMeans(k, init_method=init_method)
        indices = model.fit(x)

        assert indices.shape == torch.Size((100,))
        assert model.centroids.shape == torch.Size((k, 10))

    @pytest.mark.parametrize("k", [3, 5, 10])
    @pytest.mark.parametrize("init_method", ["random", "kmeans++"])
    def test_transform_shape(self, k: int, init_method: str):
        x = torch.randn(100, 10)
        model = KMeans(k, init_method=init_method)
        indices = model.fit(x)
        y = model.transform(indices)

        assert y.shape == torch.Size((100, 10))


class TestMultiKernelConvModule:
    @pytest.mark.parametrize(
        "dimension,x",
        [
            ["1d", torch.rand(8, 3, 32)],
            ["2d", torch.rand(8, 3, 32, 32)],
            ["3d", torch.rand(8, 3, 32, 32, 32)],
        ],
    )
    def test_output_shape(self, dimension: str, x: torch.Tensor):
        model = MultiKernelConvModule(
            3,
            10,
            kernels=[3, 5, 7],
            dimension=dimension,
            concat_output=True,
        )
        y = model(x)

        assert y.shape == torch.Size((8, 3 * 10, *x.shape[2:]))

    def test_reduction_none_output_shape(self):
        x = torch.rand(8, 3, 32, 32)
        model = MultiKernelConvModule(3, 10, kernels=[3, 5, 7], reduction="none")
        y = model(x)

        assert len(y) == 3
        for ele in y:
            assert ele.shape == torch.Size((8, 10, 32, 32))

    @pytest.mark.parametrize("reduction", ["max", "mean"])
    def test_reduction_output_shape(self, reduction: str):
        x = torch.rand(8, 3, 32, 32)
        model = MultiKernelConvModule(3, 10, kernels=[3, 5, 7], reduction=reduction)
        y = model(x)

        assert len(y) == 3
        for ele in y:
            assert ele.shape == torch.Size((8, 10))

    def test_disable_padding_output_shape(self):
        x = torch.rand(8, 3, 32, 32)
        model = MultiKernelConvModule(
            3,
            10,
            kernels=[3, 5, 7],
            disable_padding=True,
        )
        y = model(x)

        assert len(y) == 3
        for ele, k in zip(y, [3, 5, 7]):
            assert ele.shape == torch.Size((8, 10, 32 - k + 1, 32 - k + 1))
