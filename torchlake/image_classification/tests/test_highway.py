import torch

from ..models.highway.model import HighwayNetwork
from ..models.highway.network import HighwayBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
OUTPUT_SIZE = 5
HIDDEN_DIM = 64
FEATURE_MAP_SIZE = 13


class TestNetwork:
    def test_highway_block_forward_shape(self):
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        layer = HighwayBlock(HIDDEN_DIM, HIDDEN_DIM * 2, 3)
        y = layer(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM * 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    def test_highway_network_forward_shape(self):
        x = torch.randn(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        model = HighwayNetwork(INPUT_CHANNEL, OUTPUT_SIZE)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
