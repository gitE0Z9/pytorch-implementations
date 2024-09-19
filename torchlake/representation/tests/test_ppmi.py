from typing import Callable
from unittest import TestCase

import torch
from parameterized import parameterized
from torchlake.common.utils.sparse import get_sparsity

from ..models.ppmi.helper import CoOccurrenceCounter
from ..models.ppmi.model import PPMI


class TestCoOccurrenceCounter(TestCase):
    def setUp(self) -> None:
        self.vocab_size = 6
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

        self.counter = CoOccurrenceCounter(self.vocab_size)
        self.counter.update_counts(self.gram, self.context)

    def test_update_counts(self):
        self.assertDictEqual(
            self.counter.counts,
            {
                (1, 2 + 0 * self.vocab_size): 2,
                (1, 3 + 1 * self.vocab_size): 2,
                (1, 4 + 2 * self.vocab_size): 2,
                (2, 3 + 0 * self.vocab_size): 1,
                (2, 4 + 1 * self.vocab_size): 1,
                (2, 5 + 2 * self.vocab_size): 1,
            },
        )

    def test_get_context_counts(self):
        self.assertDictEqual(
            self.counter.get_context_counts(),
            {
                2 + 0 * self.vocab_size: 2,
                3 + 1 * self.vocab_size: 2,
                4 + 2 * self.vocab_size: 2,
                3 + 0 * self.vocab_size: 1,
                4 + 1 * self.vocab_size: 1,
                5 + 2 * self.vocab_size: 1,
            },
        )

    @parameterized.expand(
        [
            (
                "key_by_none",
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
                "key_by_gram",
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
                "key_by_context",
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
        ]
    )
    def test_get_pair_counts(
        self,
        name: str,
        key_by: str | None,
        expected: Callable[[int], dict[tuple[int, int], int]],
    ):
        self.assertDictEqual(
            self.counter.get_pair_counts(key_by=key_by),
            expected(self.vocab_size),
        )


class TestPPMI(TestCase):
    def setUp(self) -> None:
        self.vocab_size = 3
        self.context_size = 3
        self.context_token_size = self.vocab_size * (self.context_size - 1)
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
        self.cooccur = CoOccurrenceCounter(3)
        self.cooccur.update_counts(self.gram, self.context)
        self.vocab_counts = torch.LongTensor([0, 2, 3])
        self.model = PPMI(self.vocab_size, self.context_size)

    def test_get_embedding(self):
        self.model.fit(
            self.cooccur,
            self.vocab_counts,
        )

        embedding = self.model.embedding

        # assert sparse
        self.assertTrue(embedding.is_sparse_csr)
        # assert shape
        self.assertEqual(
            embedding.shape,
            torch.Size((self.vocab_size, self.context_token_size)),
        )
        # assert sparsity
        self.assertEqual(get_sparsity(embedding), 1 - 4 / (3 * 6))

    def test_fit(self):
        self.model.fit(
            self.cooccur,
            self.vocab_counts,
        )

        # assert sparse
        self.assertTrue(self.model.embedding.is_sparse_csr)

    def test_transform(self):
        self.model.fit(
            self.cooccur,
            self.vocab_counts,
        )

        target = self.model.transform([1, 1])

        # assert sparse
        self.assertTrue(self.model.embedding.is_sparse_csr)
        self.assertEqual(target.shape, torch.Size((2, self.context_token_size)))
        # assert sparsity
        self.assertEqual(get_sparsity(target), 1 - 4 / (2 * 6))
