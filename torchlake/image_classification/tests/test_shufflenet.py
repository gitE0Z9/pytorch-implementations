from math import ceil

import pytest
import torch

from ..models.shufflenet.model import ShuffleNet
from ..models.shufflenet.network import BottleNeck, ResBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 7


class TestNetwork:
    @pytest.mark.parametrize("input_channel,output_channel", [(48, 96), (96, 96)])
    @pytest.mark.parametrize("groups", [1, 2, 3, 4, 8])
    @pytest.mark.parametrize("group_first_pointwise", [True, False])
    def test_bottleneck_forward_shape(
        self,
        input_channel: int,
        output_channel: int,
        groups: int,
        group_first_pointwise: bool,
    ):
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = BottleNeck(
            input_channel,
            output_channel,
            groups=groups,
            group_first_pointwise=group_first_pointwise,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    @pytest.mark.parametrize(
        "input_channel,output_channel,stride,groups",
        [
            (24, 144, 2, 1),
            (144, 144, 1, 1),
            (24, 200, 2, 2),
            (200, 200, 1, 2),
            (24, 240, 2, 3),
            (240, 240, 1, 3),
            (24, 272, 2, 4),
            (272, 272, 1, 4),
            (24, 384, 2, 8),
            (384, 384, 1, 8),
        ],
    )
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
    @pytest.mark.parametrize("groups", [1, 2, 3, 4, 8])
    @pytest.mark.parametrize("scale_factor", [2, 1.5, 1, 0.5, 0.25])
    def test_shufflenet_forward_shape(self, groups: int, scale_factor: float):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = ShuffleNet(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            groups=groups,
            scale_factor=scale_factor,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
