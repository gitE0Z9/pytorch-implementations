import pytest

import torch
from ..utils.bits import get_tensor_bits


@pytest.mark.parametrize(
    "x,expected",
    [
        [torch.randint(0, 4, (10,)).to(torch.uint8), 10 * 8 * 1],
        [torch.randint(0, 4, (10,)), 10 * 8 * 8],
        [torch.rand(10), 10 * 8 * 4],
    ],
)
def test_get_tensor_bits(x: torch.Tensor, expected: int):
    assert get_tensor_bits(x) == expected
