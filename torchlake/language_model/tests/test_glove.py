import pytest
import torch
from torch.testing import assert_close

from torchlake.common.schemas.nlp import NLPContext

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
CONTEXT = NLPContext(device="cpu")
WORD_FREQS = torch.rand((VOCAB_SIZE))


class TestHelper:
    def setUp(self) -> None:
        self.gram = torch.LongTensor(
            [
                [1],
                [1],
                [2],
            ]
        )

        self.context = torch.LongTensor(
            [
                [2, 3, 4, 3],
                [2, 3, 4, 1],
                [3, 4, 5, 1],
            ]
        )

    @pytest.mark.parametrize(
        "enable_distance_weighting,expected",
        (
            (
                True,
                {
                    1 * VOCAB_SIZE + 1: 0.5,
                    1 * VOCAB_SIZE + 2: 0.5 + 0.5,
                    1 * VOCAB_SIZE + 3: 1 + 0.5 + 1,
                    1 * VOCAB_SIZE + 4: 1 + 1,
                    2 * VOCAB_SIZE + 1: 0.5,
                    2 * VOCAB_SIZE + 3: 0.5,
                    2 * VOCAB_SIZE + 4: 1,
                    2 * VOCAB_SIZE + 5: 1,
                },
            ),
            (
                False,
                {
                    1 * VOCAB_SIZE + 1: 1,
                    1 * VOCAB_SIZE + 2: 2,
                    1 * VOCAB_SIZE + 3: 3,
                    1 * VOCAB_SIZE + 4: 2,
                    2 * VOCAB_SIZE + 1: 1,
                    2 * VOCAB_SIZE + 3: 1,
                    2 * VOCAB_SIZE + 4: 1,
                    2 * VOCAB_SIZE + 5: 1,
                },
            ),
        ),
    )
    def test_cooccurrence_counter_update_counts(
        self,
        enable_distance_weighting: bool,
        expected: dict[int, int | float],
    ):
        self.setUp()

        counter = CoOccurrenceCounter(
            VOCAB_SIZE,
            enable_distance_weighting=enable_distance_weighting,
        )
        counter.update_counts(self.gram, self.context)

        assert counter.counts == expected

    @pytest.mark.parametrize(
        "enable_distance_weighting,expected",
        (
            (
                True,
                {
                    1: 1,
                    2: 1,
                    3: 3,
                    4: 3,
                    5: 1,
                },
            ),
            (
                False,
                {
                    1: 2,
                    2: 2,
                    3: 4,
                    4: 3,
                    5: 1,
                },
            ),
        ),
    )
    def test_get_context_counts(
        self,
        enable_distance_weighting: bool,
        expected: dict[int, int | float],
    ):
        self.setUp()

        counter = CoOccurrenceCounter(
            VOCAB_SIZE,
            enable_distance_weighting=enable_distance_weighting,
        )
        counter.update_counts(self.gram, self.context)

        assert counter.get_context_counts() == expected

    @pytest.mark.parametrize(
        "name,key_by,enable_distance_weighting,expected",
        [
            (
                "key_by_none",
                None,
                True,
                {
                    1 * VOCAB_SIZE + 1: 0.5,
                    1 * VOCAB_SIZE + 2: 1,
                    1 * VOCAB_SIZE + 3: 2.5,
                    1 * VOCAB_SIZE + 4: 2,
                    2 * VOCAB_SIZE + 1: 0.5,
                    2 * VOCAB_SIZE + 3: 0.5,
                    2 * VOCAB_SIZE + 4: 1,
                    2 * VOCAB_SIZE + 5: 1,
                },
            ),
            (
                "key_by_gram",
                "gram",
                True,
                {
                    1: {
                        1: 0.5,
                        2: 1,
                        3: 2.5,
                        4: 2,
                    },
                    2: {
                        1: 0.5,
                        3: 0.5,
                        4: 1,
                        5: 1,
                    },
                },
            ),
            (
                "key_by_context",
                "context",
                True,
                {
                    1: {1: 0.5, 2: 0.5},
                    2: {1: 1},
                    3: {1: 2.5, 2: 0.5},
                    4: {1: 2, 2: 1},
                    5: {2: 1},
                },
            ),
            (
                "key_by_none",
                None,
                False,
                {
                    1 * VOCAB_SIZE + 1: 1,
                    1 * VOCAB_SIZE + 2: 2,
                    1 * VOCAB_SIZE + 3: 3,
                    1 * VOCAB_SIZE + 4: 2,
                    2 * VOCAB_SIZE + 1: 1,
                    2 * VOCAB_SIZE + 3: 1,
                    2 * VOCAB_SIZE + 4: 1,
                    2 * VOCAB_SIZE + 5: 1,
                },
            ),
            (
                "key_by_gram",
                "gram",
                False,
                {
                    1: {
                        1: 1,
                        2: 2,
                        3: 3,
                        4: 2,
                    },
                    2: {
                        1: 1,
                        3: 1,
                        4: 1,
                        5: 1,
                    },
                },
            ),
            (
                "key_by_context",
                "context",
                False,
                {
                    1: {1: 1, 2: 1},
                    2: {1: 2},
                    3: {1: 3, 2: 1},
                    4: {1: 2, 2: 1},
                    5: {2: 1},
                },
            ),
        ],
    )
    def test_get_pair_counts(
        self,
        name: str,
        key_by: str | None,
        enable_distance_weighting: bool,
        expected: dict[tuple[int, int], int],
    ):
        self.setUp()

        counter = CoOccurrenceCounter(
            VOCAB_SIZE,
            enable_distance_weighting=enable_distance_weighting,
        )
        counter.update_counts(self.gram, self.context)

        assert counter.get_pair_counts(key_by=key_by) == expected

    @pytest.mark.parametrize(
        "enable_distance_weighting,expected",
        (
            (
                True,
                torch.sparse_coo_tensor(
                    [[1, 1, 1, 1, 2, 2, 2, 2], [1, 2, 3, 4, 1, 3, 4, 5]],
                    [0.5, 1, 2.5, 2, 0.5, 0.5, 1, 1],
                    size=(VOCAB_SIZE, VOCAB_SIZE),
                ),
            ),
            (
                False,
                torch.sparse_coo_tensor(
                    [[1, 1, 1, 1, 2, 2, 2, 2], [1, 2, 3, 4, 1, 3, 4, 5]],
                    [1, 2, 3, 2, 1, 1, 1, 1],
                    size=(VOCAB_SIZE, VOCAB_SIZE),
                ),
            ),
        ),
    )
    def test_get_tensor(
        self,
        enable_distance_weighting: bool,
        expected: dict[int, int | float],
    ):
        self.setUp()

        counter = CoOccurrenceCounter(
            VOCAB_SIZE,
            enable_distance_weighting=enable_distance_weighting,
        )
        counter.update_counts(self.gram, self.context)

        assert_close(counter.get_tensor().coalesce(), expected)


class TestModel:
    def test_glove_forward_shape(self):
        gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        context = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))

        model = GloVe(VOCAB_SIZE, EMBED_SIZE)

        y = model.forward(gram, context)

        assert y.shape == torch.Size((BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE))


class TestLoss:
    def setUp(self) -> None:
        self.pred = torch.rand(BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE)
        self.gram = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, 1))
        self.context = torch.randint(
            0, VOCAB_SIZE, (BATCH_SIZE * SUBSEQ_LEN, NEIGHBOR_SIZE)
        )

    def test_glove_loss_build_weighted_prob_shape(self):
        self.setUp()

        counter = CoOccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        criterion = GloVeLoss(counter.get_tensor(), maximum_count=5)

        assert criterion.weighted_prob.shape == torch.Size((VOCAB_SIZE, VOCAB_SIZE))

    def test_glove_loss_index_sparse_tensor_shape(self):
        self.setUp()

        counter = CoOccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        criterion = GloVeLoss(counter.get_tensor(), maximum_count=5)

        pred = self.pred.view(-1)
        index = torch.stack(
            (
                self.gram.repeat_interleave(self.context.size(1), 1).view(-1),
                self.context.view(-1),
            ),
            1,
        )

        y = criterion._index_sparse_tensor(
            criterion.co_occurrence_counts, index, pred.shape
        )

        assert y.shape == torch.Size((BATCH_SIZE * SUBSEQ_LEN * NEIGHBOR_SIZE,))

    def test_glove_loss_forward(self):
        self.setUp()

        model = GloVe(VOCAB_SIZE, EMBED_SIZE, CONTEXT)
        counter = CoOccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        criterion = GloVeLoss(counter.get_tensor(), maximum_count=5)
        pred = model(self.gram, self.context)
        y: torch.Tensor = criterion(self.gram, self.context, pred)

        assert not y.isnan()
