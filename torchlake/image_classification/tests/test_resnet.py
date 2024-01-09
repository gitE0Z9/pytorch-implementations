import pytest
import torch

from ..models.resnet.model import ResNet


@pytest.mark.parametrize(
    "name,num_layer",
    [
        ["18", 18],
        ["34", 34],
        ["50", 50],
        ["101", 101],
        ["152", 152],
    ],
)
def test_forward_shape(name: str, num_layer: int):
    x = torch.randn(2, 3, 224, 224)
    model = ResNet(output_size=5, num_layer=num_layer)
    y = model(x)

    assert y.shape == torch.Size((2, 5))
