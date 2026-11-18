from math import ceil

import pytest
import torch

from ..models.ghostnetv2.model import GhostNetV2
from ..models.ghostnetv2.network import DFCAttention, GhostBottleNeckV2, GhostBlockV2

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 7
EXPANSION_SIZE = 128


class TestNetwork:
    @pytest.mark.parametrize(
        "horizontal_kernel,vertical_kernel", [(3, 3), (5, 5), (7, 7)]
    )
    @pytest.mark.parametrize("feature_map_size", [7, 14])
    def test_dfc_attention_forward_shape(
        self,
        horizontal_kernel: int,
        vertical_kernel: int,
        feature_map_size: int,
    ):
        x = torch.randn(BATCH_SIZE, 64, feature_map_size, feature_map_size)
        model = DFCAttention(64, 96, horizontal_kernel, vertical_kernel)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, 96, feature_map_size, feature_map_size)
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
    def test_ghost_bottleneck_v2_forward_shape(
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
        model = GhostBottleNeckV2(
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
    def test_ghost_block_v2_forward_shape(
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
        model = GhostBlockV2(
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
    def test_ghostnet_v2_forward_shape(self, width_multiplier: float):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = GhostNetV2(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            width_multiplier=width_multiplier,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
