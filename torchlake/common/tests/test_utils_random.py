import torch
from torchlake.common.utils.random import generate_normal


def test_shape():
    x = generate_normal(1, 5)

    assert x.shape == torch.Size((1, 5))
