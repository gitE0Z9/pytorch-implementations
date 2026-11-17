import pytest
import torch

from ..models.densenet.model import DenseNet
from ..models.densenet.network import DenseBlock, TransitionBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
HIDDEN_DIM = 16
FEATURE_MAP_SIZE = 13
GROWTH_RATE = 32


class TestNetwork:
    def test_denseblock_forward_shape(self):
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        layer = DenseBlock(HIDDEN_DIM, 4, growth_rate=GROWTH_RATE)
        y = layer(x)

        assert y.shape == torch.Size(
            (
                BATCH_SIZE,
                HIDDEN_DIM + 4 * GROWTH_RATE,
                FEATURE_MAP_SIZE,
                FEATURE_MAP_SIZE,
            )
        )

    def test_transitionblock_forward_shape(self):
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        layer = TransitionBlock(HIDDEN_DIM, HIDDEN_DIM // 2)
        y = layer(x)

        assert y.shape == torch.Size(
            (
                BATCH_SIZE,
                HIDDEN_DIM // 2,
                FEATURE_MAP_SIZE // 2,
                FEATURE_MAP_SIZE // 2,
            )
        )


class TestModel:
    @pytest.mark.parametrize("key", [121, 169, 201, 264])
    def test_densenet_forward_shape(self, key: int):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = DenseNet(INPUT_CHANNEL, OUTPUT_SIZE, key=key)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
