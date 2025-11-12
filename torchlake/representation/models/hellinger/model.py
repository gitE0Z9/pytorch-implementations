from pathlib import Path

import joblib
import torch
import torch.nn.functional as F
from torch import nn
from torchlake.common.models import KernelPCA
from torchlake.common.models.kernel_pca import KernelEnum

from .helper import CooccurrenceCounter


class HellingerPCA(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        maximum_context_size: int = 10000,
        n_components: int = 50,
        # enable_incremental_pca: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_components = min(vocab_size, n_components)
        self.maximum_context_size = min(vocab_size, maximum_context_size)
        self.model = KernelPCA(self.n_components, kernel=KernelEnum.HELLINGER)

    @property
    def embedding(self) -> torch.Tensor:
        return self.model.eigenvectors

    def fit(self, co_occurrence: CooccurrenceCounter, vocab_counts: torch.LongTensor):
        # build co-occurrence matrix
        count_source = co_occurrence.get_pair_counts().items()
        row_indices = []
        col_indices = []
        values = []
        for (gram, context), count in count_source:
            row_indices.append(gram)
            col_indices.append(context)
            values.append(count)
        counts = torch.sparse_coo_tensor(
            [row_indices, col_indices],
            values,
            size=(self.vocab_size, self.vocab_size),
        ).to(vocab_counts.device)

        # select most significant context words
        most_significant_context = vocab_counts.topk(self.maximum_context_size).indices
        counts: torch.Tensor = counts.index_select(1, most_significant_context)
        prob = counts.to_dense().float()
        self.model.fit(F.normalize(prob, 1, 1))

    def transform(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding[tokens]

    def save(self, path: str | Path):
        joblib.dump(self.model, path)

    def load(self, path: str | Path):
        self.model = joblib.load(path)
