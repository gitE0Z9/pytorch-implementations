import pytest
import torch

from ..models.mobilenet.model import MobileNetV1


@pytest.mark.parametrize(
    "name,resolution_multiplier,width_multiplier",
    [
        ["rho=224,alpha=1", 224, 1],
        ["rho=224,alpha=0.75", 224, 0.75],
        ["rho=224,alpha=0.5", 224, 0.5],
        ["rho=224,alpha=0.25", 224, 0.25],
        ["rho=128,alpha=1", 128, 1],
        ["rho=128,alpha=0.75", 128, 0.75],
        ["rho=128,alpha=0.5", 128, 0.5],
        ["rho=128,alpha=0.25", 128, 0.25],
    ],
)
def test_mobilenetv1_forward_shape(
    name: str,
    resolution_multiplier: int,
    width_multiplier: float,
):
    x = torch.randn(2, 3, resolution_multiplier, resolution_multiplier)
    model = MobileNetV1(output_size=5, width_multiplier=width_multiplier)
    y = model(x)

    assert y.shape == torch.Size((2, 5))
