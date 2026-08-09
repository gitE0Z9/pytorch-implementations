import torch
from torch import nn
from torchlake.common.utils.sparse import ones_tensor


class GloVeLoss(nn.Module):

    def __init__(
        self,
        co_occurrence_counts: torch.Tensor,
        alpha: float = 0.75,
        maximum_count: int = 100,
    ):
        """GloVe loss function

        Args:
            co_occurrence_counts (torch.Tensor): a sparse coo tensor, shape is (vocab_size, vocab_size), value is word-word cooccurrence counts
            alpha (float, optional): power of the weighted probability. Defaults to 0.75.
            maximum_count (int, optional): maximum occurrence count as cutoff of the weighted probability. Defaults to 100.
        """
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")
        self.co_occurrence_counts = co_occurrence_counts.coalesce()
        self.weighted_prob = self._build_weighted_prob(alpha, maximum_count)

    def _build_weighted_prob(self, alpha: float, maximum_count: int) -> torch.Tensor:
        """build the weighted probability

        Args:
            alpha (float, optional): power of the weighted probability. Defaults to 0.75.
            maximum_count (int, optional): maximum occurrence count as cutoff of the weighted probability. Defaults to 100.

        Returns:
            torch.Tensor: the weighted probability, a sparse coo tensor, shape is (vocab_size, vocab_size)
        """
        v = self.co_occurrence_counts.values()
        values = torch.where(
            v < maximum_count,
            (v / maximum_count) ** alpha,
            torch.ones_like(v, dtype=torch.float),
        )

        return torch.sparse_coo_tensor(
            self.co_occurrence_counts.indices(),
            values,
            size=self.co_occurrence_counts.size(),
        ).coalesce()

    def _index_sparse_tensor(
        self,
        ref: torch.Tensor,
        index: torch.Tensor,
        output_shape: torch.Size,
    ) -> torch.Tensor:
        """retrive from sparse tensor by 2d indices

        Args:
            ref (torch.Tensor): sparse tensor to be selected, shape is (vocab_size, vocab_size)
            index (torch.Tensor): shape is (batch_size*subseq_len*neighbor_size, 2)
            output_shape: shape of output tensor

        Returns:
            torch.Tensor: tensor from sparse tensor by indices, shape is (batch_size*subseq_len*neighbor_size,)
        """
        vocab_size = ref.shape[0]
        key = ref.indices()
        key = key[0] * vocab_size + key[1]

        query = index[:, 0] * vocab_size + index[:, 1]

        pos = torch.searchsorted(key, query).clamp(max=key.numel() - 1)
        hit = key[pos] == query

        value = torch.zeros(query.shape, dtype=ref.dtype, device=ref.device)
        value[hit] = ref.values()[pos[hit]]

        return value.reshape(output_shape)

    def forward(
        self,
        gram: torch.Tensor,
        context: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """calculate GloVe loss value

        Args:
            gram (torch.Tensor): shape is batch_size*subseq_len, 1
            context (torch.Tensor): shape is batch_size*subseq_len, neighbor_size
            pred (torch.Tensor): prediction of GloVe, shape is batch_size*subseq_len, neighbor_size

        Returns:
            torch.Tensor: GloVe loss value
        """
        pred = pred.view(-1)
        # batch_size*subseq_len*neighbor_size, 2
        index = torch.stack(
            (
                gram.repeat_interleave(context.size(1), 1).view(-1),
                context.view(-1),
            ),
            1,
        )

        gt = self._index_sparse_tensor(self.co_occurrence_counts, index, pred.shape)
        w = self._index_sparse_tensor(self.weighted_prob, index, pred.shape)

        # log1p in on the page 4
        y = self.loss(pred, gt.log1p())
        return (y * w).mean()
