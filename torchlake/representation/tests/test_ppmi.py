import math

import torch

from torchlake.common.utils.sparse import get_sparsity

from ..models.ppmi.helper import CooccurrenceCounter
from ..models.ppmi.model import PPMI

VOCAB_SIZE = 6
CONTEXT_SIZE = 3
NEIGHBOR_SIZE = CONTEXT_SIZE - 1
LATENT_DIM = VOCAB_SIZE * NEIGHBOR_SIZE


class TestModel:
    def setup_ppmi(self) -> None:
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

    def test_ppmi_get_embedding(self):
        self.setup_ppmi()

        counter = CooccurrenceCounter(VOCAB_SIZE, NEIGHBOR_SIZE)
        counter.update_counts(self.gram, self.context)
        model = PPMI(VOCAB_SIZE, CONTEXT_SIZE)

        model.fit(counter, self.vocab_counts)

        embedding = model.embedding

        # assert sparse
        assert embedding.is_sparse_csr
        # assert shape
        assert embedding.shape == torch.Size((VOCAB_SIZE, LATENT_DIM))
        # assert sparsity
        assert get_sparsity(embedding) == 1 - 4 / math.prod(embedding.shape)

    def test_ppmi_fit(self):
        self.setup_ppmi()

        counter = CooccurrenceCounter(VOCAB_SIZE, NEIGHBOR_SIZE)
        counter.update_counts(self.gram, self.context)
        model = PPMI(VOCAB_SIZE, CONTEXT_SIZE)

        model.fit(counter, self.vocab_counts)

        # assert sparse
        assert model.embedding.is_sparse_csr

    def test_ppmi_transform(self):
        self.setup_ppmi()

        counter = CooccurrenceCounter(VOCAB_SIZE, NEIGHBOR_SIZE)
        counter.update_counts(self.gram, self.context)
        model = PPMI(VOCAB_SIZE, CONTEXT_SIZE)

        model.fit(counter, self.vocab_counts)

        target = model.transform([1, 1])

        # assert sparse
        assert model.embedding.is_sparse_csr
        assert target.shape == torch.Size((2, LATENT_DIM))
        # assert sparsity
        assert get_sparsity(model.embedding) == 1 - 4 / math.prod(model.embedding.shape)
