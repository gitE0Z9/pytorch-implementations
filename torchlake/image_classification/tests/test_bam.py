import pytest
import torch

from ..models.bam.model import BAMResNet
from ..models.bam.network import BAM2d

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
HIDDEN_DIM = 16
FEATURE_MAP_SIZE = 56


class TestNetwork:
    def test_bam2d_forward_shape(self):
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        model = BAM2d(HIDDEN_DIM, 16)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("key", [18, 34, 50, 101, 152])
    @pytest.mark.parametrize("pre_activation", [False, True])
    def test_bam_resnet_forward_shape(self, key: int, pre_activation: bool):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = BAMResNet(
            INPUT_CHANNEL,
            OUTPUT_SIZE,
            key=key,
            pre_activation=pre_activation,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
