import torch
from ..utils.sparse import eye_matrix, ones_tensor


def test_eye_matrix():
    x = eye_matrix(5)

    assert x.shape == torch.Size((5, 5))


def test_ones_tensor():
    x = torch.arange(5)
    y = ones_tensor(x.repeat(2, 1))

    assert y.shape == torch.Size((5, 5))
