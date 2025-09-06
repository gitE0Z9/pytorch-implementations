import torch

from ..models.fast_style_transfer import FastStyleTransfer
from ..models.fast_style_transfer.network import ResidualBlock


class TestNetwork:
    def test_residual_block_forward_shape(self):
        x = torch.rand((1, 3, 32, 32))

        model = ResidualBlock(3)

        y = model(x)

        assert y.shape == torch.Size((1, 3, 32, 32))


class TestModel:
    def test_forward_shape(self):
        x = torch.rand((1, 3, 32, 32))

        model = FastStyleTransfer(3)

        y = model(x)

        assert y.shape == torch.Size((1, 3, 32, 32))
