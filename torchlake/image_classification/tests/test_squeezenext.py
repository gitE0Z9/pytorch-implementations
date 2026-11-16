from math import ceil

import pytest
import torch

from ..models.squeezenext.model import SqueezeNeXt
from ..models.squeezenext.network import BottleNeck

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 13


class TestNetwork:
    @pytest.mark.parametrize(
        "input_channel,output_channel,stride",
        [
            [64, 32, 2],
            [32, 32, 1],
        ],
    )
    def test_bottleneck_forward_shape(
        self,
        input_channel: int,
        output_channel: int,
        stride: int,
    ):
        OUTPUT_SIZE = ceil(FEATURE_MAP_SIZE / stride)

        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = BottleNeck(input_channel, output_channel, stride)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, OUTPUT_SIZE, OUTPUT_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("width_multiplier", [1, 1.5, 2])
    @pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
    def test_squeezenext_forward_shape(self, width_multiplier: float, version: int):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = SqueezeNeXt(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            width_multiplier=width_multiplier,
            version=version,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
