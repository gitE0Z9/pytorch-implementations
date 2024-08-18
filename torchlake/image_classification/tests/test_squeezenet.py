import pytest
import torch

from ..models.squeezenet.model import SqueezeNet
from ..models.squeezenet.network import FireModule


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_squeezenet_forward_shape(version: str):
    x = torch.randn(2, 3, 224, 224)
    model = SqueezeNet(output_size=5, version=version)
    y = model(x)

    assert y.shape == torch.Size((2, 5))


@pytest.mark.parametrize(
    "in_c,squeeze_ratio,expand_ratio,out_c",
    [
        [96, 1 / 6, 4, 128],
        [128, 1 / 8, 4, 128],
    ],
)
def test_fire_module_forward_shape(
    in_c: int,
    squeeze_ratio: float,
    expand_ratio: float,
    out_c: int,
):
    x = torch.randn(2, in_c, 7, 7)
    model = FireModule(in_c, squeeze_ratio, expand_ratio)
    y = model(x)

    assert y.shape == torch.Size((2, out_c, 7, 7))
