import torch

from ..models.parsenet import ParseNet, GlobalContextModule


class TestParseNet:
    def test_forward_shape(self):
        x = torch.rand((1, 3, 32, 32))

        model = ParseNet(21)

        y = model(x)

        assert y.shape == torch.Size((1, 21, 32, 32))


class TestGlobalContextModule:
    def test_forward_shape(self):
        x = torch.rand((1, 3, 32, 32))

        model = GlobalContextModule(3, 3)

        y = model(x)

        assert y.shape == torch.Size((1, 3, 32, 32))
