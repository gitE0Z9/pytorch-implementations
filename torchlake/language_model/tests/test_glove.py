from unittest import TestCase

import torch
from parameterized import parameterized
from torch.testing import assert_close
from torchlake.common.schemas.nlp import NlpContext

from ..models.glove.helper import CoOccurrenceCounter
from ..models.glove.loss import GloVeLoss
from ..models.glove.model import GloVe

BATCH_SIZE = 2
VOCAB_SIZE = 16
CONTEXT_SIZE = 5
EMBED_SIZE = 8
NEIGHBOR_SIZE = CONTEXT_SIZE - 1
SUBSEQ_LEN = 256 - NEIGHBOR_SIZE
NEGATIVE_RATIO = 5
CONTEXT = NlpContext(device="cpu")
WORD_FREQS = torch.rand((VOCAB_SIZE))


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


class TestModel(TestCase):
    def test_forward_shape(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))

        model = GloVe(VOCAB_SIZE, EMBED_SIZE)

        y = model.forward(gram, context)

        self.assertEqual(y.shape, torch.Size((BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE)))


class TestLoss(TestCase):
    def setUp(self) -> None:
        self.gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        self.context = torch.randint(
            0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE)
        )
        counter = CoOccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)
        self.counts = counter.get_tensor()

    def test_build_weighted_prob_shape(self):
        criterion = GloVeLoss(self.counts, maximum_count=1)

        assert criterion.weighted_prob.shape == torch.Size((VOCAB_SIZE, VOCAB_SIZE))

    def test_index_sparse_tensor_shape(self):
        criterion = GloVeLoss(self.counts, maximum_count=1)

        context_shape = self.context.shape
        gram = self.gram.repeat_interleave(self.context.size(1), 1).view(-1)
        context = self.context.view(-1)

        y = criterion.index_sparse_tensor(self.counts, gram, context, context_shape)

        assert y.shape == torch.Size((BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))

    def test_forward_shape(self):
        model = GloVe(VOCAB_SIZE, EMBED_SIZE, CONTEXT)
        criterion = GloVeLoss(self.counts, maximum_count=1)

        pred = model.forward(self.gram, self.context)
        y: torch.Tensor = criterion.forward(self.gram, self.context, pred)

        assert not y.isnan()
