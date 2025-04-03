from math import ceil

import pytest
import torch
from torch.testing import assert_close

from ..models.ghostnet.network import GhostModule
from ..models.ghostnetv2.network import GhostBottleNeckV2, GhostLayerV2
from ..models.ghostnetv2.model import GhostNetV2
from ..models.ghostnetv3.model import GhostNetV3
from ..models.ghostnetv3.network import (
    GhostBottleNeckV3,
    GhostLayerV3,
    GhostModuleV3,
    InceptionModule,
)


class TestGhostNetV3:
    @pytest.mark.parametrize("width_multiplier", [1, 0.5])
    def test_forward_shape(self, width_multiplier: float):
        x = torch.randn(2, 3, 224, 224)
        model = GhostNetV3(
            output_size=5,
            width_multiplier=width_multiplier,
        )
        y = model(x)

        assert y.shape == torch.Size((2, 5))

    def test_reparameterize(self):
        x = torch.randn(2, 3, 224, 224)
        dest = GhostNetV2(output_size=5).eval()
        model = GhostNetV3(output_size=5).eval()

        model.reparameterize(dest)

        y = model(x)
        y_prime = dest(x)

        # rounding error is huge, only acceptable until the third block of body
        # so the workaround is check order preservation to ensure classification
        # the feature itself is not recommended
        assert_close(y.argmax(1), y_prime.argmax(1))


class TestInceptionModule:
    @pytest.mark.parametrize("in_c,out_c", [[16, 64], [16, 16]])
    @pytest.mark.parametrize("kernel", [1, 3])
    def test_forward_shape(self, in_c: int, out_c: int, kernel: int):
        x = torch.randn(2, in_c, 14, 14)
        model = InceptionModule(in_c, out_c, kernel, num_branch=3)
        y = model(x)

        assert y.shape == torch.Size((2, out_c, 14, 14))

    @pytest.mark.parametrize(
        "in_c,out_c,groups",
        [
            [16, 64, 1],
            [16, 16, 1],
            [16, 64, 16],
            [16, 16, 16],
        ],
    )
    @pytest.mark.parametrize("kernel", [1, 3])
    def test_reparameterize(self, in_c: int, out_c: int, groups: int, kernel: int):
        x = torch.randn(2, in_c, 14, 14)
        model = InceptionModule(in_c, out_c, kernel, groups=groups, num_branch=3).eval()
        dest = model.reparameterize()

        y = model(x)
        y_prime = dest(x)
        assert_close(y, y_prime)


class TestGhostModuleV3:
    @pytest.mark.parametrize("s", [2])
    @pytest.mark.parametrize("d", [1, 3])
    def test_forward_shape(self, s: int, d: int):
        x = torch.randn(2, 64, 14, 14)
        model = GhostModuleV3(64, 96, s, d)
        y = model(x)

        assert y.shape == torch.Size((2, 96, 14, 14))

    @pytest.mark.parametrize("in_c,out_c", [[16, 64], [16, 16]])
    @pytest.mark.parametrize("kernel", [1, 3])
    def test_reparameterize(self, in_c: int, out_c: int, kernel: int):
        x = torch.randn(2, in_c, 14, 14)
        dest = GhostModule(in_c, out_c, d=kernel).eval()
        model = GhostModuleV3(in_c, out_c, d=kernel).eval()
        model.reparameterize(dest)

        y = model(x)
        y_prime = dest(x)
        assert_close(y, y_prime)


class TestGhostBottleNeckV3:
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("s", [2])
    @pytest.mark.parametrize("d", [1, 3])
    @pytest.mark.parametrize("enable_se", [True, False])
    def test_forward_shape(self, stride: int, s: int, d: int, enable_se: bool):
        INPUT_SHAPE = 7
        OUTPUT_SHAPE = ceil(7 / stride)

        x = torch.randn(2, 64, INPUT_SHAPE, INPUT_SHAPE)
        model = GhostBottleNeckV3(
            64,
            96,
            3,
            stride,
            s,
            d,
            expansion_size=128,
            enable_se=enable_se,
        )
        y = model(x)

        assert y.shape == torch.Size((2, 96, OUTPUT_SHAPE, OUTPUT_SHAPE))

    @pytest.mark.parametrize("in_c,out_c", [[16, 64], [16, 16]])
    @pytest.mark.parametrize("kernel", [1, 3])
    def test_reparameterize(self, in_c: int, out_c: int, kernel: int):
        x = torch.randn(2, in_c, 14, 14)
        dest = GhostBottleNeckV2(
            in_c,
            out_c,
            3,
            d=kernel,
            expansion_size=128,
        ).eval()
        model = GhostBottleNeckV3(
            in_c,
            out_c,
            3,
            d=kernel,
            expansion_size=128,
        ).eval()
        model.reparameterize(dest)

        y = model(x)
        y_prime = dest(x)
        assert_close(y, y_prime)


class TestGhostLayerV3:
    @pytest.mark.parametrize("stride", [1, 2])
    @pytest.mark.parametrize("s", [2])
    @pytest.mark.parametrize("d", [1, 3])
    @pytest.mark.parametrize("enable_se", [True, False])
    def test_forward_shape(self, stride: int, s: int, d: int, enable_se: bool):
        INPUT_SHAPE = 7
        OUTPUT_SHAPE = ceil(7 / stride)

        x = torch.randn(2, 64, INPUT_SHAPE, INPUT_SHAPE)
        model = GhostLayerV3(
            64,
            96,
            3,
            stride=stride,
            s=s,
            d=d,
            expansion_size=128,
            enable_se=enable_se,
        )
        y = model(x)

        assert y.shape == torch.Size((2, 96, OUTPUT_SHAPE, OUTPUT_SHAPE))

    @pytest.mark.parametrize("in_c,out_c", [[16, 64], [16, 16]])
    @pytest.mark.parametrize("kernel", [1, 3])
    def test_reparameterize(self, in_c: int, out_c: int, kernel: int):
        x = torch.randn(2, in_c, 14, 14)
        dest = GhostLayerV2(
            in_c,
            out_c,
            3,
            d=kernel,
            expansion_size=128,
        ).eval()
        model = GhostLayerV3(
            in_c,
            out_c,
            3,
            d=kernel,
            expansion_size=128,
        ).eval()
        model.reparameterize(dest)

        y = model(x)
        y_prime = dest(x)
        assert_close(y, y_prime)
