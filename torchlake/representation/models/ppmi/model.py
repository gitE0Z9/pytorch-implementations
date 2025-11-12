from functools import cached_property

import torch
from torch import nn
from tqdm import tqdm

from .helper import CooccurrenceCounter


class PPMI(nn.Module):

    def __init__(self, vocab_size: int, context_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.row_indices = []
        self.col_indices = []
        self.values = []

    @property
    def embed_dim(self) -> int:
        return (self.context_size - 1) * self.vocab_size

    @cached_property
    def embedding(self) -> torch.Tensor:
        return nn.Parameter(
            torch.sparse_coo_tensor(
                [self.row_indices, self.col_indices],
                self.values,
                (self.vocab_size, self.embed_dim),
            ).to_sparse_csr(),
            requires_grad=False,
        )

    def fit(
        self,
        co_occurrence: CooccurrenceCounter,
        vocab_counts: torch.LongTensor,
        show_progress: bool = True,
    ):
        corpus_total = vocab_counts.sum()
        context_counts = co_occurrence.get_context_counts()

        count_source = co_occurrence.get_pair_counts().items()
        if show_progress:
            count_source = tqdm(count_source)
        for (gram, context), pair_count in count_source:
            self.row_indices.append(gram)
            self.col_indices.append(context)

            ppmi = torch.log2(
                corpus_total * pair_count / context_counts[context] / vocab_counts[gram]
            )
            self.values.append(ppmi.clip(0).item())

    def transform(self, tokens: list[int]) -> torch.Tensor:
        if self.embedding is None:
            raise ValueError("The model has not been fitted yet.")

        return torch.stack([self.embedding[token] for token in tokens])
