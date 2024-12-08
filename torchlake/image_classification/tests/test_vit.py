import torch

from ..models.vit.model import ViT


def test_forward_shape():
    x = torch.rand((2, 3, 32, 32))

    model = ViT(3, 10, image_size=32, patch_size=16)

    y = model(x)

    assert y.shape == torch.Size((2, 10))


def test_eval_forward_shape():
    x = torch.rand((2, 3, 64, 64))

    model = ViT(3, 10, image_size=32, patch_size=16)

    y = model(x)

    assert y.shape == torch.Size((2, 10))
