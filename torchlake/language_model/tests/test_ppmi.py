from unittest import TestCase

import torch
from torchlake.common.utils.sparse import get_sparsity

from ..models.ppmi.helper import CoOccurrenceCounter
from ..models.ppmi.model import PPMI


class TestCoOccurrenceCounter(TestCase):
    def setUp(self) -> None:
        self.gram = torch.Tensor(
            [
                [1],
                [1],
                [2],
            ]
        )

        self.context = torch.Tensor(
            [
                [2, 3, 4],
                [2, 3, 4],
                [3, 4, 5],
            ]
        )

    def test_update_counts(self):
        counter = CoOccurrenceCounter()
        counter.update_counts(self.gram, self.context)

        self.assertDictEqual(
            counter.counts,
            {
                (1, (2, 3, 4)): 2,
                (2, (3, 4, 5)): 1,
            },
        )

    def test_get_context_counts(self):
        counter = CoOccurrenceCounter()
        counter.update_counts(self.gram, self.context)

        self.assertDictEqual(
            counter.get_context_counts(),
            {
                (2, 3, 4): 2,
                (3, 4, 5): 1,
            },
        )

    def test_get_pair_counts_key_by_none(self):
        counter = CoOccurrenceCounter()
        counter.update_counts(self.gram, self.context)

        self.assertDictEqual(
            counter.get_pair_counts(),
            {
                (1, (2, 3, 4)): 2,
                (2, (3, 4, 5)): 1,
            },
        )

    def test_get_pair_counts_key_by_gram(self):
        counter = CoOccurrenceCounter()
        counter.update_counts(self.gram, self.context)

        self.assertDictEqual(
            counter.get_pair_counts(key_by="gram"),
            {
                1: {
                    (2, 3, 4): 2,
                },
                2: {
                    (3, 4, 5): 1,
                },
            },
        )

    def test_get_pair_counts_key_by_context(self):
        counter = CoOccurrenceCounter()
        counter.update_counts(self.gram, self.context)

        self.assertDictEqual(
            counter.get_pair_counts(key_by="context"),
            {
                (2, 3, 4): {
                    1: 2,
                },
                (3, 4, 5): {
                    2: 1,
                },
            },
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
        self.cooccur = CoOccurrenceCounter()
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
        self.assertEqual(get_sparsity(embedding), 1 - 2 / (3 * 6))

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
        self.assertEqual(get_sparsity(target), 1 - 2 / (2 * 6))
