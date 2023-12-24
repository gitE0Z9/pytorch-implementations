import torch

from ..models import PrototypicalNetwork


def test_feature_shape():
    x = torch.randn(2, 1, 28, 28)
    model = PrototypicalNetwork(1)
    y = model.feature_extract(x)

    assert y.shape == torch.Size((2, 64))
