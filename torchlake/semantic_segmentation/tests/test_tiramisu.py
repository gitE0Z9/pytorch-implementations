import pytest
import torch

from ..models.tiramisu.model import fc_densenet_56, fc_densenet_67, fc_densenet_103
from ..models.tiramisu.network import DenseBlock, TransitionDown

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 512
FEATURE_MAP_SIZE_8X = IMAGE_SIZE // 8
FEATURE_MAP_SIZE_16X = IMAGE_SIZE // 16
NUM_CLASS = 21
NUM_LAYER = 4
GROWTH_RATE = 12


class TestNetwork:
    def test_dense_block_forward_shape(self):
        x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE_16X, FEATURE_MAP_SIZE_16X)
        )

        model = DenseBlock(HIDDEN_DIM, GROWTH_RATE, num_layer=NUM_LAYER)
        y = model(x)

        assert y.shape == torch.Size(
            (
                BATCH_SIZE,
                HIDDEN_DIM + NUM_LAYER * GROWTH_RATE,
                FEATURE_MAP_SIZE_16X,
                FEATURE_MAP_SIZE_16X,
            )
        )

    def test_transition_down_forward_shape(self):
        x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE_8X, FEATURE_MAP_SIZE_8X)
        )

        model = TransitionDown(HIDDEN_DIM)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE_16X, FEATURE_MAP_SIZE_16X)
        )


class TestModel:
    @pytest.mark.parametrize(
        "constructor", (fc_densenet_56, fc_densenet_67, fc_densenet_103)
    )
    def test_fc_densenet_forward_shape(self, constructor):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = constructor(INPUT_CHANNEL, NUM_CLASS)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
