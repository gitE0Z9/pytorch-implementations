from math import ceil

import pytest
import torch

from ..models.shufflenetv2.model import ShuffleNetV2
from ..models.shufflenetv2.network import BottleNeck, ResBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 7


class TestNetwork:
    @pytest.mark.parametrize("input_channel,output_channel", [(48, 96), (96, 96)])
    def test_bottleneck_forward_shape(self, input_channel: int, output_channel: int):
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = BottleNeck(input_channel, output_channel)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    @pytest.mark.parametrize(
        "input_channel,output_channel,stride",
        [
            (24, 48, 2),
            (48, 48, 1),
            (24, 116, 2),
            (116, 116, 1),
            (24, 176, 2),
            (176, 176, 1),
            (24, 244, 2),
            (244, 244, 1),
        ],
    )
    @pytest.mark.parametrize("groups", [1, 2])
    def test_resblock_forward_shape(
        self,
        input_channel: int,
        output_channel: int,
        stride: int,
        groups: int,
    ):
        OUTPUT_SIZE = ceil(FEATURE_MAP_SIZE / stride)

        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = ResBlock(
            input_channel,
            output_channel,
            stride=stride,
            groups=groups,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, OUTPUT_SIZE, OUTPUT_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("scale_factor", [2, 1.5, 1, 0.5])
    def test_shufflenetv2_forward_shape(self, groups: int, scale_factor: float):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = ShuffleNetV2(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            groups=groups,
            scale_factor=scale_factor,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
