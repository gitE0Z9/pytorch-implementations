import pytest
import torch
from torch import nn
from torch.testing import assert_close

from ..helpers.fuser import (
    fuse_conv_bn,
    fuse_sum_parallel_convs,
    fuse_concat_parallel_convs,
    fuse_sequential_convs,
    convert_bn_to_conv,
)


class TestFuseConv:
    def setUp(self, in_c: int = 3):
        self.x = torch.rand(2, in_c, 7, 7)

    @pytest.mark.parametrize(
        "in_c,out_c,groups",
        [
            [3, 6, 1],
            [6, 6, 1],
            [3, 6, 3],
            [6, 6, 6],
        ],
    )
    @pytest.mark.parametrize("kernel", [1, 3, 5])
    def test_conv_bn(self, in_c: int, out_c: int, groups: int, kernel: int):
        self.setUp(in_c)

        layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, 1, kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
        ).eval()
        layer = nn.Conv2d(in_c, out_c, kernel, 1, kernel // 2, groups=groups)

        fuse_conv_bn(*layers, dest=layer)

        y = layers(self.x)
        y_prime = layer(self.x)

        assert_close(y, y_prime)

    @pytest.mark.parametrize(
        "in_c,out_c,groups",
        [
            [6, 6, 1],
            # [6, 6, 2],
            # [6, 6, 6],
        ],
    )
    @pytest.mark.parametrize("kernel", [1, 3, 5])
    def test_bn_to_conv(self, in_c: int, out_c: int, groups: int, kernel: int):
        self.setUp(in_c)

        with torch.no_grad():
            bn = nn.BatchNorm2d(out_c).eval()
            layer = nn.Conv2d(in_c, out_c, kernel, 1, kernel // 2, groups=groups)

            convert_bn_to_conv(bn, layer)

            y = bn(self.x)
            y_prime = layer(self.x)

        assert_close(y, y_prime)

    @pytest.mark.parametrize(
        "in_c,out_c,groups",
        [
            [3, 6, 1],
            [6, 6, 1],
            [3, 6, 3],
            [6, 6, 6],
        ],
    )
    @pytest.mark.parametrize("kernels", [[1, 3], [3, 1], [1, 1], [3, 3]])
    @pytest.mark.parametrize("bias", [True, False])
    def test_sequential_convs(
        self,
        in_c: int,
        out_c: int,
        groups: int,
        kernels: list[int],
        bias: bool,
    ):
        self.setUp(in_c)

        layers = nn.Sequential(
            *[
                nn.Conv2d(
                    in_c if i == 0 else out_c,
                    out_c,
                    kernel,
                    1,
                    kernel // 2,
                    bias=bias,
                    groups=groups,
                )
                for i, kernel in enumerate(kernels)
            ]
        )
        layer = nn.Conv2d(
            in_c,
            out_c,
            max(kernels),
            1,
            max(kernels) // 2,
            groups=groups,
            bias=bias,
        )

        fuse_sequential_convs(*layers, dest=layer)

        y = layers(self.x)
        y_prime = layer(self.x)

        assert_close(y, y_prime)

    @pytest.mark.parametrize(
        "in_c,out_c,groups",
        [
            [3, 6, 1],
            [6, 6, 1],
            [3, 6, 3],
            [6, 6, 6],
        ],
    )
    @pytest.mark.parametrize("kernels", [[1, 1, 1], [1, 3], [3, 3, 3]])
    @pytest.mark.parametrize("bias", [True, False])
    def test_sum_parallel_convs(
        self,
        in_c: int,
        out_c: int,
        groups: int,
        kernels: list[int],
        bias: bool,
    ):
        self.setUp(in_c)

        layers = [
            nn.Conv2d(in_c, out_c, kernel, 1, kernel // 2, groups=groups, bias=bias)
            for kernel in kernels
        ]
        layer = nn.Conv2d(
            in_c, out_c, max(kernels), 1, max(kernels) // 2, groups=groups
        )

        fuse_sum_parallel_convs(*layers, dest=layer)

        y = sum(layer(self.x) for layer in layers)
        y_prime = layer(self.x)

        assert_close(y, y_prime)

    @pytest.mark.parametrize(
        "in_c,out_c,groups",
        [
            [3, 6, 1],
            [6, 6, 1],
            # [3, 6, 3],
            # [6, 6, 6],
        ],
    )
    @pytest.mark.parametrize("kernels", [[1, 1, 1], [1, 3], [3, 3, 3]])
    @pytest.mark.parametrize("bias", [True, False])
    def test_concat_parallel_convs(
        self,
        in_c: int,
        out_c: int,
        groups: int,
        kernels: list[int],
        bias: bool,
    ):
        self.setUp(in_c)

        layers = [
            nn.Conv2d(in_c, out_c, kernel, 1, kernel // 2, groups=groups, bias=bias)
            for kernel in kernels
        ]
        layer = nn.Conv2d(
            in_c,
            out_c * len(kernels),
            max(kernels),
            1,
            max(kernels) // 2,
            # groups=groups,
        )

        fuse_concat_parallel_convs(*layers, dest=layer)

        y = torch.cat([layer(self.x) for layer in layers], 1)
        y_prime = layer(self.x)

        assert_close(y, y_prime)
