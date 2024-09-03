from ..utils.text import get_unigram_counts, get_unigram_counts_by_tensor
import random
import torch
from torch.testing import assert_close


def test_get_unigram_counts():
    X = torch.arange(0, 1000).view(1, -1).repeat(32, 1)

    counts = get_unigram_counts(X.tolist(), 1000)

    assert_close(counts, torch.ones((1000), dtype=torch.long) * 32)


def test_get_unigram_counts_by_tensor():
    X = torch.arange(0, 1000).view(1, -1).repeat(32, 1)

    counts = get_unigram_counts_by_tensor(X, 1000)

    assert_close(counts, torch.ones((1000), dtype=torch.long) * 32)
