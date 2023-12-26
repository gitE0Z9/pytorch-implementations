import torch
import pytest
from ..utils.numerical import safe_sqrt, safe_negative_factorial


@pytest.mark.parametrize(
    "name,x",
    [
        ["zero", torch.zeros((1))],
        ["random", torch.rand((1))],
    ],
)
def test_safe_sqrt(name, x):
    y = safe_sqrt(x)

    assert not torch.isnan(y).any().item()


@pytest.mark.parametrize(
    "name,x",
    [
        ["zero", torch.zeros((1))],
        ["random", torch.rand((1))],
    ],
)
def test_safe_negative_factorial(name, x):
    y = safe_negative_factorial(x, -0.5)

    assert not torch.isnan(y).any().item()
