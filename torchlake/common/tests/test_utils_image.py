import torch
from ..utils.image import yiq_transform, yiq_inverse_transform


def test_yiq_transform():
    x = torch.rand((1, 3, 16, 16))
    y = yiq_transform(x)

    assert y.shape == torch.Size((1, 3, 16, 16))


def test_yiq_inverse_transform():
    x = torch.rand((1, 3, 16, 16))
    y = yiq_inverse_transform(x)

    assert y.shape == torch.Size((1, 3, 16, 16))
