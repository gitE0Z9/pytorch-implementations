import torch
from torchlake.common.utils.random import generate_standard_normal


def test_shape():
    x = generate_standard_normal(1, 5)

    assert x.shape == torch.Size((1, 5))
