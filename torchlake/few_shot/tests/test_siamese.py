import torch

from ..models import SiameseNetwork


def test_forward_shape():
    x, x2 = torch.randn(2, 1, 105, 105), torch.randn(2, 1, 105, 105)
    model = SiameseNetwork()
    y = model(x, x2)

    assert y.shape == torch.Size(2, 1)
