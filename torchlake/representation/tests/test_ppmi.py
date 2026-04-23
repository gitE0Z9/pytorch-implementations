from typing import Callable
from unittest import TestCase

import torch
import pytest
from torchlake.common.utils.sparse import get_sparsity

from ..models.ppmi.helper import CooccurrenceCounter
from ..models.ppmi.model import PPMI

VOCAB_SIZE = 6


class TestCooccurrenceCounter:
    def setup_cooccurrence_counter(self) -> None:
        self.gram = torch.LongTensor(
            [
                [1],
                [1],
                [2],
            ]
        )

        self.context = torch.LongTensor(
            [
                [2, 3, 4],
                [2, 3, 4],
                [3, 4, 5],
            ]
        )

    def test_update_counts_cooccurrence_counter(self):
        self.setup_cooccurrence_counter()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        assert counter.counts == {
            (1, 2 + 0 * VOCAB_SIZE): 2,
            (1, 3 + 1 * VOCAB_SIZE): 2,
            (1, 4 + 2 * VOCAB_SIZE): 2,
            (2, 3 + 0 * VOCAB_SIZE): 1,
            (2, 4 + 1 * VOCAB_SIZE): 1,
            (2, 5 + 2 * VOCAB_SIZE): 1,
        }

    def test_get_context_counts_cooccurrence_counter(self):
        self.setup_cooccurrence_counter()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        assert counter.get_context_counts() == {
            2 + 0 * VOCAB_SIZE: 2,
            3 + 1 * VOCAB_SIZE: 2,
            4 + 2 * VOCAB_SIZE: 2,
            3 + 0 * VOCAB_SIZE: 1,
            4 + 1 * VOCAB_SIZE: 1,
            5 + 2 * VOCAB_SIZE: 1,
        }

    @pytest.mark.parametrize(
        "key_by,expected",
        (
            (
                None,
                lambda vocab_size: {
                    (1, 2 + 0 * vocab_size): 2,
                    (1, 3 + 1 * vocab_size): 2,
                    (1, 4 + 2 * vocab_size): 2,
                    (2, 3 + 0 * vocab_size): 1,
                    (2, 4 + 1 * vocab_size): 1,
                    (2, 5 + 2 * vocab_size): 1,
                },
            ),
            (
                "gram",
                lambda vocab_size: {
                    1: {
                        2 + 0 * vocab_size: 2,
                        3 + 1 * vocab_size: 2,
                        4 + 2 * vocab_size: 2,
                    },
                    2: {
                        3 + 0 * vocab_size: 1,
                        4 + 1 * vocab_size: 1,
                        5 + 2 * vocab_size: 1,
                    },
                },
            ),
            (
                "context",
                lambda vocab_size: {
                    2 + 0 * vocab_size: {1: 2},
                    3 + 1 * vocab_size: {1: 2},
                    4 + 2 * vocab_size: {1: 2},
                    3 + 0 * vocab_size: {2: 1},
                    4 + 1 * vocab_size: {2: 1},
                    5 + 2 * vocab_size: {2: 1},
                },
            ),
        ),
    )
    def test_get_pair_counts_cooccurrence_counter(
        self,
        key_by: str | None,
        expected: Callable[[int], dict[tuple[int, int], int]],
    ):
        self.setup_cooccurrence_counter()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        assert counter.get_pair_counts(key_by=key_by) == expected(VOCAB_SIZE)


class TestModel:
    def setup_ppmi(self) -> None:
        self.context_size = 3
        self.context_token_size = VOCAB_SIZE * (self.context_size - 1)
        self.gram = torch.LongTensor(
            [
                [1],
                [2],
                [1],
            ]
        )

        self.context = torch.LongTensor(
            [
                [2, 2],
                [1, 1],
                [2, 2],
            ]
        )

        self.vocab_counts = torch.LongTensor([0, 2, 3, 0, 0, 0])

    def test_get_embedding_ppmi(self):
        self.setup_ppmi()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)
        model = PPMI(VOCAB_SIZE, self.context_size)

        model.fit(counter, self.vocab_counts)

        embedding = model.embedding

        # assert sparse
        assert embedding.is_sparse_csr
        # assert shape
        assert embedding.shape == torch.Size((VOCAB_SIZE, self.context_token_size))
        # assert sparsity
        assert get_sparsity(embedding) == 1 - 4 / (6 * 12)

    def test_fit_ppmi(self):
        self.setup_ppmi()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)
        model = PPMI(VOCAB_SIZE, self.context_size)

        model.fit(counter, self.vocab_counts)

        # assert sparse
        assert model.embedding.is_sparse_csr

    def test_transform_ppmi(self):
        self.setup_ppmi()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)
        model = PPMI(VOCAB_SIZE, self.context_size)

        model.fit(counter, self.vocab_counts)

        target = model.transform([1, 1])

        # assert sparse
        assert model.embedding.is_sparse_csr
        assert target.shape == torch.Size((2, self.context_token_size))
        # assert sparsity
        assert get_sparsity(model.embedding) == 1 - 4 / (6 * 12)
