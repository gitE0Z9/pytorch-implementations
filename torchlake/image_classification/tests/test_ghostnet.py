from math import ceil

import pytest
import torch

from ..models.ghostnet.model import GhostNet
from ..models.ghostnet.network import GhostBottleNeck, GhostModule

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 7
EXPANSION_SIZE = 128


class TestNetwork:
    @pytest.mark.parametrize("d", [1, 3])
    def test_ghost_module_forward_shape(self, d: int):
        x = torch.randn(BATCH_SIZE, 64, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = GhostModule(64, 96, num_ghost_feature=2, kernel_ds=d)
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
    def test_ghost_bottleneck_forward_shape(
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
        model = GhostBottleNeck(
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
    def test_ghostnet_forward_shape(self, width_multiplier: float):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = GhostNet(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            width_multiplier=width_multiplier,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
