import torch

from ..models.tcn.model import Tcn


def test_forward_shape_model():
    x = torch.rand((2, 3, 1024))
    model = Tcn(3, num_channels=[1, 32, 64, 128, 1])
    y = model(x)

    assert y.shape == torch.Size((2, 1, 1024))
