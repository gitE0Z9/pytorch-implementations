import pytest
import torch
from torch.testing import assert_close

from ..utils.text import get_context, get_unigram_counts, get_unigram_counts_by_tensor


def test_get_unigram_counts():
    X = torch.arange(0, 1000).view(1, -1).repeat(32, 1)

    counts = get_unigram_counts(X.tolist(), 1000)

    assert_close(counts, torch.ones((1000), dtype=torch.long) * 32)


def test_get_unigram_counts_by_tensor():
    X = torch.arange(0, 1000).view(1, -1).repeat(32, 1)

    counts = get_unigram_counts_by_tensor(X, 1000)

    assert_close(counts, torch.ones((1000), dtype=torch.long) * 32)


@pytest.mark.parametrize(
    "name,left_context_size,right_context_size,enable_random_context_size,enable_symmetric_context,expected_gram,expected_context",
    [
        [
            "both size 2",
            2,
            2,
            False,
            True,
            [3, 4, 5],
            [[1, 2, 4, 5], [2, 3, 5, 6], [3, 4, 6, 7]],
        ],
        [
            "random max size 2",
            2,
            2,
            True,
            True,
            [3, 4, 5],
            [[1, 2, 4, 5], [2, 3, 5, 6], [3, 4, 6, 7]],
        ],
        [
            "prev and next 1",
            0,
            1,
            False,
            False,
            [1, 2, 3, 4, 5, 6],
            [[2], [3], [4], [5], [6], [7]],
        ],
        [
            "prev 5 and no more",
            5,
            0,
            False,
            False,
            [6, 7],
            [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
        ],
    ],
)
def test_get_context(
    name: str,
    left_context_size: int,
    right_context_size: int,
    enable_random_context_size: bool,
    enable_symmetric_context: bool,
    expected_gram: list[int],
    expected_context: list[list[int]],
):
    X = [torch.LongTensor([1, 2, 3, 4, 5, 6, 7])] * 32

    gram, context = get_context(
        X,
        left_context_size,
        right_context_size,
        enable_random_context_size,
        enable_symmetric_context,
    )

    assert_close(gram, torch.LongTensor(expected_gram).repeat(32, 1).unsqueeze(1))

    if not enable_random_context_size:
        assert_close(
            context,
            torch.LongTensor(expected_context)
            .transpose(-1, -2)
            .unsqueeze(0)
            .repeat(32, 1, 1),
        )
    else:
        assert context.size(0) == 32
        assert 0 < context.size(1) <= left_context_size + right_context_size
        assert context.size(2) == len(expected_gram)
