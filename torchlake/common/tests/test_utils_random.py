import torch
from torchlake.common.utils.random import generate_normal, generate_uniform


def test_generate_normal_shape():
    x = generate_normal(1, 5)

    assert x.shape == torch.Size((1, 5))


def test_generate_uniform_shape():
    x = generate_uniform(1, 5)

    assert x.shape == torch.Size((1, 5))
