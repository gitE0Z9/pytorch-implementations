import pytest
import torch

from ..models.fast_style_transfer import FastStyleTransfer
from ..models.fast_style_transfer.network import ConvBlock, ResidualBlock


class TestConvBlock:
    def test_forward_shape(self):
        x = torch.rand((1, 3, 32, 32))

        model = ConvBlock(3, 8, 3)

        y = model(x)

        assert y.shape == torch.Size((1, 8, 32, 32))


class TestResidualBlock:
    def test_forward_shape(self):
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
