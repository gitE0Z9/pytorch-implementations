from operator import itemgetter

import torch
from torch import nn

from .helper import CooccurrenceCounter


class PPMI(nn.Module):

    def __init__(self, vocab_size: int, context_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding = None

    @property
    def embed_dim(self) -> int:
        return (self.context_size - 1) * self.vocab_size

    def fit(self, counter: CooccurrenceCounter, vocab_counts: torch.LongTensor):
        device = vocab_counts.device

        corpus_total = vocab_counts.sum()
        context_counts = counter.get_context_counts()

        count_source: torch.Tensor = counter.get_tensor().coalesce().to(device)
        indices, pair_count = count_source.indices(), count_source.values()

        norminator = corpus_total * pair_count
        denominator = vocab_counts[indices[0]] * torch.tensor(
            itemgetter(*indices[1].tolist())(context_counts),
            device=device,
        )
        ppmi = torch.log2(norminator / denominator)

        self.embedding = nn.Parameter(
            torch.sparse_coo_tensor(
                indices,
                ppmi.clip(0, 1000),
                (self.vocab_size, self.embed_dim),
            ).to_sparse_csr(),
            requires_grad=False,
        )

    def transform(self, tokens: list[int]) -> torch.Tensor:
        if self.embedding is None:
            raise ValueError("The model has not been fitted yet.")

        return torch.stack([self.embedding[token] for token in tokens])
