import torch

from torchlake.common.helpers.counter import CooccurrenceCounter

from ..models.hellinger.model import HellingerPCA

VOCAB_SIZE = 6


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
