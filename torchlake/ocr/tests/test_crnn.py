import torch
from ..models.crnn.model import Crnn


def test_forward_shape():
    x = torch.randn(1, 3, 32, 1698)
    model = Crnn(3, 8, 10)
    output = model(x)

    assert output.shape == torch.Size((1698 // 4 - 1, 1, 10))
