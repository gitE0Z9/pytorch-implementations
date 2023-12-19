import torch
from ..models import UNet


def test_forward_shape():
    x = torch.rand((1, 3, 32, 32))

    model = UNet(21)

    y = model(x)

    assert y.shape == torch.Size((1, 21, 32, 32))
