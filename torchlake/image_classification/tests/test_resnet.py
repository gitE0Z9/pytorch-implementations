import torch

from ..models import ResNet50


def test_forward_shape():
    x = torch.randn(2, 3, 224, 224)
    model = ResNet50(5)
    y = model(x)

    assert y.shape == torch.Size(2, 5)
