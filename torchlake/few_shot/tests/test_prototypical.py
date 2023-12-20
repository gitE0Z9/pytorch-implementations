import torch

from ..models import PrototypicalNetwork


def test_forward_shape():
    x = torch.randn(2, 1, 105, 105)
    model = PrototypicalNetwork()
    y = model(x)

    assert y.shape == torch.Size(2, 256, 3 * 3)
