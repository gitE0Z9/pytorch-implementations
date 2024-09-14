import torch
from ..utils.sparse import eye_matrix, ones_tensor, get_sparsity


def test_eye_matrix():
    x = eye_matrix(5)

    assert x.shape == torch.Size((5, 5))


def test_ones_tensor():
    x = torch.arange(5)
    y = ones_tensor(x.repeat(2, 1))

    assert y.shape == torch.Size((5, 5))


def test_get_sparsity():
    x = torch.sparse_coo_tensor([[0, 1], [1, 1]], [5, 6], (2, 2))
    y = get_sparsity(x)

    assert y == 2 / 4
