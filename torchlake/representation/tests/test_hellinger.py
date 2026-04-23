import pytest
import torch
from torch.testing import assert_close

from ..models.hellinger.helper import CooccurrenceCounter
from ..models.hellinger.model import HellingerPCA

VOCAB_SIZE = 6


class TestHelper:
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
            (1, 2): 2,
            (1, 3): 2,
            (1, 4): 2,
            (2, 3): 1,
            (2, 4): 1,
            (2, 5): 1,
        }

    def test_get_context_counts_cooccurrence_counter(self):
        self.setup_cooccurrence_counter()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        assert counter.get_context_counts() == {
            2: 2,
            3: 3,
            4: 3,
            5: 1,
        }

    @pytest.mark.parametrize(
        "key_by,expected",
        (
            (
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
                "context",
                {
                    2: {1: 2},
                    3: {1: 2, 2: 1},
                    4: {1: 2, 2: 1},
                    5: {2: 1},
                },
            ),
        ),
        # ids=(None, "gram", "context"),
    )
    def test_get_pair_counts_cooccurrence_counter(
        self,
        key_by: str | None,
        expected: dict[tuple[int, int], int],
    ):
        self.setup_cooccurrence_counter()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        assert counter.get_pair_counts(key_by=key_by) == expected

    def test_get_tensor_cooccurrence_counter(self):
        self.setup_cooccurrence_counter()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        assert_close(
            counter.get_tensor(),
            torch.sparse_coo_tensor(
                [[1, 1, 1, 2, 2, 2], [2, 3, 4, 3, 4, 5]],
                [2, 2, 2, 1, 1, 1],
                size=(VOCAB_SIZE, VOCAB_SIZE),
            ),
        )


class TestModel:
    def setup_hellinger_pca(self) -> None:
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

        self.vocab_counts = torch.LongTensor([0, 2, 3, 0, 0, 0])

    def test_fit_hellinger_pca(self):
        self.setup_hellinger_pca()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        model = HellingerPCA(VOCAB_SIZE)

        model.fit(counter, self.vocab_counts)

        assert hasattr(model.model, "eigenvectors")

    def test_embedding_hellinger_pca(self):
        self.setup_hellinger_pca()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        model = HellingerPCA(VOCAB_SIZE)

        model.fit(counter, self.vocab_counts)

        embedding = model.embedding

        assert embedding.shape == torch.Size((VOCAB_SIZE, model.n_components))

    def test_transform_hellinger_pca(self):
        self.setup_hellinger_pca()

        counter = CooccurrenceCounter(VOCAB_SIZE)
        counter.update_counts(self.gram, self.context)

        model = HellingerPCA(VOCAB_SIZE)

        model.fit(counter, self.vocab_counts)

        target = model.transform([1, 1])

        assert target.shape == torch.Size((2, model.n_components))
