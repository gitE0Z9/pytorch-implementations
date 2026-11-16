import pytest
import torch

from ..models.squeezenet.model import SqueezeNet
from ..models.squeezenet.network import FireModule

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 7


class TestNetwork:
    @pytest.mark.parametrize(
        "input_channel,squeeze_ratio,expand_ratio,output_channel",
        [
            [96, 1 / 6, 4, 128],
            [128, 1 / 8, 4, 128],
        ],
    )
    def test_fire_module_forward_shape(
        self,
        input_channel: int,
        squeeze_ratio: float,
        expand_ratio: float,
        output_channel: int,
    ):
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = FireModule(input_channel, squeeze_ratio, expand_ratio)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("version", ["1.0", "1.1"])
    def test_squeezenet_forward_shape(self, version: str):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = SqueezeNet(INPUT_CHANNEL, OUTPUT_SIZE, version=version)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
