import pytest
import torch

from ..models.vovnet.model import VoVNet
from ..models.vovnet.network import OSAModule

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
FEATURE_MAP_SIZE = 7


class TestNetwork:
    @pytest.mark.parametrize(
        "input_channel,output_channel",
        [
            [64, 128],
            [128, 128],
        ],
    )
    def test_osa_module_forward_shape(
        self,
        input_channel: int,
        output_channel: int,
    ):
        x = torch.randn(BATCH_SIZE, input_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = OSAModule(
            input_channel,
            32,
            output_channel,
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, output_channel, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("key", ["27_slim", 39, 57])
    def test_vovnet_forward_shape(self, key: str | int):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = VoVNet(INPUT_CHANNEL, OUTPUT_SIZE, key=key)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
