import torch
import pytest
from ..utils.numerical import safe_sqrt, safe_negative_factorial, generate_grid
from torch.testing import assert_close


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


def test_generate_grid():
    x, y = generate_grid(3, 3)

    assert_close(
        x,
        torch.LongTensor(
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
            ]
        ),
    )

    assert_close(
        y,
        torch.LongTensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
            ]
        ),
    )
