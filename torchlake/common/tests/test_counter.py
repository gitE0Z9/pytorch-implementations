import pytest
import torch
from torch.testing import assert_close

from ..helpers.counter import CooccurrenceCounter

VOCAB_SIZE = 6


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

    def test_cooccurrence_counter_update_counts(self):
        self.setUp()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        assert counter.counts == {
            1 * VOCAB_SIZE + 1: 1,
            1 * VOCAB_SIZE + 2: 2,
            1 * VOCAB_SIZE + 3: 3,
            1 * VOCAB_SIZE + 4: 2,
            2 * VOCAB_SIZE + 1: 1,
            2 * VOCAB_SIZE + 3: 1,
            2 * VOCAB_SIZE + 4: 1,
            2 * VOCAB_SIZE + 5: 1,
        }

    def test_cooccurrence_counter_get_context_counts(self):
        self.setUp()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        assert counter.get_context_counts() == {
            1: 2,
            2: 2,
            3: 4,
            4: 3,
            5: 1,
        }

    @pytest.mark.parametrize(
        "key_by,expected",
        (
            (
                None,
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
                "gram",
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
                "context",
                {
                    1: {1: 1, 2: 1},
                    2: {1: 2},
                    3: {1: 3, 2: 1},
                    4: {1: 2, 2: 1},
                    5: {2: 1},
                },
            ),
        ),
        # ids=(None, "gram", "context"),
    )
    def test_cooccurrence_counter_get_pair_counts(
        self,
        key_by: str | None,
        expected: dict[tuple[int, int], int],
    ):
        self.setUp()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        assert counter.get_pair_counts(key_by=key_by) == expected

    def test_cooccurrence_counter_get_tensor(self):
        self.setUp()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        assert_close(
            counter.get_tensor().coalesce(),
            torch.sparse_coo_tensor(
                [[1, 1, 1, 1, 2, 2, 2, 2], [1, 2, 3, 4, 1, 3, 4, 5]],
                [1, 2, 3, 2, 1, 1, 1, 1],
                size=(VOCAB_SIZE, VOCAB_SIZE),
            ),
        )
