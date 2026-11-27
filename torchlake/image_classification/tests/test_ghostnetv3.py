from math import ceil

import pytest
import torch
from torch.testing import assert_close

from ..models.ghostnet.network import GhostModule
from ..models.ghostnetv2.model import GhostNetV2
from ..models.ghostnetv2.network import GhostBlockV2, GhostBottleNeckV2
from ..models.ghostnetv3.model import GhostNetV3
from ..models.ghostnetv3.network import (
    GhostBlockV3,
    GhostBottleNeckV3,
    GhostModuleV3,
)

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 7
EXPANSION_SIZE = 128
NUM_BRANCH = 3


class TestNetwork:
    @pytest.mark.parametrize("kernel_ds", [1, 3])
    def test_ghost_module_v3_forward_shape(self, kernel_ds: int):
        x = torch.randn(BATCH_SIZE, 64, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = GhostModuleV3(64, 96, num_ghost_feature=2, kernel_ds=kernel_ds)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, 96, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    @pytest.mark.parametrize(
        "name,input_channel,output_channel,stride",
        [
            ["first", 64, 96, 2],
            ["middle", 96, 96, 1],
        ],
    )
    @pytest.mark.parametrize("kernel_ds", [1, 3])
    @pytest.mark.parametrize("enable_se", [True, False])
    def test_ghost_bottleneck_v3_forward_shape(
        self,
        name: str,
        input_channel: int,
        output_channel: int,
        stride: int,
        kernel_ds: int,
        enable_se: bool,
    ):
        OUTPUT_SHAPE = ceil(FEATURE_MAP_SIZE / stride)

        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = GhostBottleNeckV3(
            input_channel,
            output_channel,
            3,
            stride=stride,
            num_ghost_feature=2,
            kernel_ds=kernel_ds,
            expansion_size=EXPANSION_SIZE,
            enable_se=enable_se,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, OUTPUT_SHAPE, OUTPUT_SHAPE)
        )

    @pytest.mark.parametrize(
        "name,input_channel,output_channel,stride",
        [
            ["first", 64, 96, 2],
            ["middle", 96, 96, 1],
        ],
    )
    @pytest.mark.parametrize("kernel_ds", [1, 3])
    @pytest.mark.parametrize("enable_se", [True, False])
    def test_ghost_block_v3_forward_shape(
        self,
        name: str,
        input_channel: int,
        output_channel: int,
        stride: int,
        kernel_ds: int,
        enable_se: bool,
    ):
        OUTPUT_SHAPE = ceil(FEATURE_MAP_SIZE / stride)

        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = GhostBlockV3(
            input_channel,
            output_channel,
            3,
            stride=stride,
            num_ghost_feature=2,
            kernel_ds=kernel_ds,
            expansion_size=EXPANSION_SIZE,
            enable_se=enable_se,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, OUTPUT_SHAPE, OUTPUT_SHAPE)
        )


class TestModel:
    @pytest.mark.parametrize("width_multiplier", [1, 0.5])
    def test_ghostnet_v3_forward_shape(self, width_multiplier: float):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = GhostNetV3(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            width_multiplier=width_multiplier,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))


class TestReparameterize:
    @pytest.mark.parametrize("input_channel,output_channel", [[16, 64], [16, 16]])
    @pytest.mark.parametrize("kernel_ds", [1, 3])
    def test_ghost_module_v3(
        self, input_channel: int, output_channel: int, kernel_ds: int
    ):
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        dest = GhostModule(input_channel, output_channel, kernel_ds=kernel_ds).eval()
        model = GhostModuleV3(input_channel, output_channel, kernel_ds=kernel_ds).eval()
        model.reparameterize(dest)

        assert_close(model(x), dest(x))

    @pytest.mark.parametrize(
        "input_channel,output_channel,stride",
        [
            [16, 64, 2],
            [16, 64, 1],
            [16, 16, 1],
        ],
    )
    @pytest.mark.parametrize("kernel_ds", [1, 3])
    def test_ghost_bottleneck_v3(
        self,
        input_channel: int,
        output_channel: int,
        stride: int,
        kernel_ds: int,
    ):
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        dest = GhostBottleNeckV2(
            input_channel,
            output_channel,
            3,
            stride=stride,
            kernel_ds=kernel_ds,
            expansion_size=EXPANSION_SIZE,
        ).eval()
        model = GhostBottleNeckV3(
            input_channel,
            output_channel,
            3,
            stride=stride,
            kernel_ds=kernel_ds,
            expansion_size=EXPANSION_SIZE,
        ).eval()
        model.reparameterize(dest)

        assert_close(model(x), dest(x))

    @pytest.mark.parametrize(
        "input_channel,output_channel,stride",
        [
            [16, 64, 2],
            [16, 64, 1],
            [16, 16, 1],
        ],
    )
    @pytest.mark.parametrize("kernel_ds", [1, 3])
    def test_ghost_block_v3(
        self,
        input_channel: int,
        output_channel: int,
        stride: int,
        kernel_ds: int,
    ):
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        dest = GhostBlockV2(
            input_channel,
            output_channel,
            3,
            stride=stride,
            kernel_ds=kernel_ds,
            expansion_size=EXPANSION_SIZE,
        ).eval()
        model = GhostBlockV3(
            input_channel,
            output_channel,
            3,
            stride=stride,
            kernel_ds=kernel_ds,
            expansion_size=EXPANSION_SIZE,
        ).eval()
        model.reparameterize(dest)

        assert_close(model(x), dest(x))

    def test_ghostnet_v3(self):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        dest = GhostNetV2(INPUT_CHANNEL, OUTPUT_SIZE).eval()
        model = GhostNetV3(INPUT_CHANNEL, OUTPUT_SIZE).eval()
        model.reparameterize(dest)

        assert_close(model(x), dest(x))

    def test_ghostnet_v3_reparameterized_load_state_dict(self):
        """reparameterized weights saved as tensors only (e.g. safetensors) can be loaded into a fresh GhostNetV2"""
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        dest = GhostNetV2(INPUT_CHANNEL, OUTPUT_SIZE).eval()
        GhostNetV3(INPUT_CHANNEL, OUTPUT_SIZE).eval().reparameterize(dest)

        model = GhostNetV2(INPUT_CHANNEL, OUTPUT_SIZE).eval()
        model.load_state_dict(dest.state_dict(), strict=True)

        assert_close(model(x), dest(x))
