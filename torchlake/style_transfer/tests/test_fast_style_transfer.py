import torch

from ..models.fast_style_transfer import FastStyleTransfer
from ..models.fast_style_transfer.network import ResidualBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 32
HIDDEN_DIM = 8


class TestNetwork:
    def test_residual_block_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))

        model = ResidualBlock(HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))


class TestModel:
    def test_fast_style_transfer_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = FastStyleTransfer(3)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        )
