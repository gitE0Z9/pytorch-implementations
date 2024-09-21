from unittest import TestCase

import torch
from parameterized import parameterized
from torchlake.common.utils.sparse import get_sparsity

from ..models.hellinger.helper import CoOccurrenceCounter
from ..models.hellinger.model import HellingerPCA
from torch.testing import assert_close


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
                (1, 2): 2,
                (1, 3): 2,
                (1, 4): 2,
                (2, 3): 1,
                (2, 4): 1,
                (2, 5): 1,
            },
        )

    def test_get_context_counts(self):
        self.assertDictEqual(
            self.counter.get_context_counts(),
            {
                2: 2,
                3: 3,
                4: 3,
                5: 1,
            },
        )

    @parameterized.expand(
        [
            (
                "key_by_none",
                None,
                {
                    (1, 2): 2,
                    (1, 3): 2,
                    (1, 4): 2,
                    (2, 3): 1,
                    (2, 4): 1,
                    (2, 5): 1,
                },
            ),
            (
                "key_by_gram",
                "gram",
                {
                    1: {
                        2: 2,
                        3: 2,
                        4: 2,
                    },
                    2: {
                        3: 1,
                        4: 1,
                        5: 1,
                    },
                },
            ),
            (
                "key_by_context",
                "context",
                {
                    2: {1: 2},
                    3: {1: 2, 2: 1},
                    4: {1: 2, 2: 1},
                    5: {2: 1},
                },
            ),
        ]
    )
    def test_get_pair_counts(
        self,
        name: str,
        key_by: str | None,
        expected: dict[tuple[int, int], int],
    ):
        self.assertDictEqual(
            self.counter.get_pair_counts(key_by=key_by),
            expected,
        )

    def test_get_tensor(self):
        assert_close(
            self.counter.get_tensor(),
            torch.sparse_coo_tensor(
                [[1, 1, 1, 2, 2, 2], [2, 3, 4, 3, 4, 5]],
                [2, 2, 2, 1, 1, 1],
                size=(self.vocab_size, self.vocab_size),
            ),
        )


class TestHellingerPCA(TestCase):
    def setUp(self) -> None:
        self.vocab_size = 3
        self.context_size = 3
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
        self.model = HellingerPCA(self.vocab_size)

    def test_fit(self):
        self.model.fit(
            self.cooccur,
            self.vocab_counts,
        )

        self.assertTrue(hasattr(self.model.model, "eigenvectors"))

    def test_get_embedding(self):
        self.model.fit(
            self.cooccur,
            self.vocab_counts,
        )

        embedding = self.model.get_embedding()

        # assert shape
        self.assertEqual(
            embedding.shape,
            torch.Size((self.vocab_size, self.model.n_components)),
        )

    def test_transform(self):
        self.model.fit(
            self.cooccur,
            self.vocab_counts,
        )

        target = self.model.transform([1, 1])

        self.assertEqual(target.shape, torch.Size((2, self.model.n_components)))
