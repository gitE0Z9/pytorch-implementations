import torch
from torch import nn

from torchlake.common.helpers.counter import CooccurrenceCounter
from torchlake.common.models import KernelPCA
from torchlake.common.models.kernel_pca import KernelEnum


class HellingerPCA(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        maximum_context_size: int = 10000,
        n_components: int = 50,
        # enable_incremental_pca: bool = False,
        enable_sparse_svd: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_components = min(vocab_size, n_components)
        self.maximum_context_size = min(vocab_size, maximum_context_size)
        self.model = KernelPCA(
            self.n_components,
            kernel=KernelEnum.HELLINGER,
            enable_sparse_svd=enable_sparse_svd,
        )

    @property
    def embedding(self) -> torch.Tensor:
        output = self.model.eigen_vectors
        assert output is not None, "please fit first."
        return output

    def fit(
        self,
        counter: CooccurrenceCounter,
        vocab_counts: torch.LongTensor,
        show_progress: bool = True,
    ):
        counts = counter.get_tensor(device=vocab_counts.device).coalesce()

        most_significant_context = vocab_counts.topk(self.maximum_context_size).indices
        # vocab_size, maximum_context_size
        counts: torch.Tensor = counts.index_select(1, most_significant_context)
        self.model.fit(counts.float().to_dense(), show_progress=show_progress)

    def transform(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding[tokens]
