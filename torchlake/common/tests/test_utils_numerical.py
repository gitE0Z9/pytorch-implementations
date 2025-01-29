import pytest
import torch
from torch.testing import assert_close

from ..utils.numerical import (
    build_gaussian_heatmap,
    generate_grid,
    log_sum_exp,
    safe_negative_factorial,
    safe_sqrt,
)


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


def test_log_sum_exp():
    x = torch.rand(2, 10)
    y = log_sum_exp(x)

    assert y.shape == torch.Size((2,))
    assert not torch.isnan(y).any()


@pytest.mark.parametrize(
    "shape,center,expected",
    [
        (
            (3,),
            False,
            (torch.LongTensor([0, 1, 2]),),
        ),
        (
            (3,),
            True,
            (torch.LongTensor([0, 1, 2]) - 1,),
        ),
        (
            (3, 3),
            False,
            (
                torch.LongTensor([0, 1, 2]).expand(3, 3),
                torch.LongTensor([0, 1, 2]).expand(3, 3).mT,
            ),
        ),
        (
            (3, 3),
            True,
            (
                torch.LongTensor([0, 1, 2]).expand(3, 3) - 1,
                torch.LongTensor([0, 1, 2]).expand(3, 3).mT - 1,
            ),
        ),
        (
            (3, 3, 3),
            False,
            (
                torch.LongTensor([0, 1, 2]).expand(3, 3, 3).mT,
                torch.LongTensor([0, 1, 2]).expand(3, 3, 3).permute(2, 1, 0),
                torch.LongTensor([0, 1, 2]).expand(3, 3, 3),
            ),
        ),
        (
            (3, 3, 3),
            True,
            (
                torch.LongTensor([0, 1, 2]).expand(3, 3, 3).mT - 1,
                torch.LongTensor([0, 1, 2]).expand(3, 3, 3).permute(2, 1, 0) - 1,
                torch.LongTensor([0, 1, 2]).expand(3, 3, 3) - 1,
            ),
        ),
    ],
)
def test_generate_grid(shape: tuple[int], center: bool, expected: tuple[torch.Tensor]):
    grids = generate_grid(*shape, center=center)

    for grid, y in zip(grids, expected):
        assert_close(grid, y)


@pytest.mark.parametrize("truncated", [False, True])
def test_gaussian_heatmap(truncated):
    x = torch.Tensor(
        [
            [
                [1, 1],
                [1, 2],
            ]
        ],
    )
    y = build_gaussian_heatmap(
        x,
        (5, 5),
        sigma=1,
        truncated=truncated,
    )

    assert_close(y.shape, torch.Size((1, 2, 5, 5)))
    assert y[0, 0, 1, 1] == y[0, 0].max()
    assert y[0, 1, 1, 2] == y[0, 1].max()
