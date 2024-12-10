import torch
from ..models.diracnet.model import DiracNet
from ..models.diracnet.network import DiracConv2d


class TestDiracNet:
    def test_forward_shape(self):
        x = torch.rand((1, 3, 32, 32))

        model = DiracNet(output_size=10)

        y = model(x)

        assert y.shape == torch.Size((1, 10))


class TestDiracConv2d:
    def test_forward_shape(self):
        x = torch.rand((1, 3, 32, 32))

        model = DiracConv2d(input_channel=3, output_channel=5, kernel=3)

        y = model(x)

        assert y.shape == torch.Size((1, 5, 32, 32))
