import pytest
import torch
from ..models.pixelcnn.model import PixelCNN
from ..models.pixelcnn.network import MaskedConv2d

BATCH_SIZE = 2
IMAGE_SIZE = 32
HIDDEN_DIM = 8
OUTPUT_SIZE = 5


class TestNetwork:
    @pytest.mark.parametrize("mask_type", ["A", "B"])
    def test_forward_shape(self, mask_type: str):
        x = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

        model = MaskedConv2d(3, HIDDEN_DIM, 3, padding=1, mask_type=mask_type)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))


class TestModel:
    def test_forward_shape(self):
        x = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)

        model = PixelCNN(1, 256, HIDDEN_DIM, 6)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, 256, IMAGE_SIZE, IMAGE_SIZE))
