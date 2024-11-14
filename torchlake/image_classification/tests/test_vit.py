import torch
from ..models.vit.model import ViT
import pytest


@pytest.mark.parametrize("y", [torch.randint(0, 10, (2,)), None])
def test_forward_shape(y: torch.Tensor | None):
    x = torch.rand((2, 3, 32, 32))

    model = ViT(3, 10, int(32 / 4))

    y = model(x, y)

    assert y.shape == torch.Size((2, 10))
