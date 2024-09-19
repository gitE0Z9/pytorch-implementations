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
        super(GloVeLoss, self).__init__()
        self.loss = nn.MSELoss(reduction="none")
        self.co_occurrence_counts = co_occurrence_counts.coalesce()
        self.weighted_prob = self._build_weighted_prob(alpha, maximum_count)

    def _build_weighted_prob(self, alpha: float, maximum_count: int) -> torch.Tensor:
        """build the weighted probability

        Args:
            alpha (float, optional): power of the weighted probability. Defaults to 0.75.
            maximum_count (int, optional): maximum occurrence count as cutoff of the weighted probability. Defaults to 100.

        Returns:
            torch.Tensor: the weighted probability, a sparse coo tensor
        """
        v = self.co_occurrence_counts.values()
        values = torch.ones_like(v, dtype=torch.float)
        values[v < maximum_count].div_(maximum_count).pow_(alpha)

        return torch.sparse_coo_tensor(
            self.co_occurrence_counts.indices(),
            values,
            size=self.co_occurrence_counts.size(),
        )

    def index_sparse_tensor(
        self,
        target: torch.Tensor,
        gram: torch.Tensor,
        context: torch.Tensor,
        context_shape,
    ) -> torch.Tensor:
        """get tensor from sparse tensor by indices

        Args:
            gram (torch.Tensor): shape is batch_size*subseq_len*neighbor_size
            context (torch.Tensor): shape is batch_size*subseq_len*neighbor_size

        Returns:
            torch.Tensor: tensor from sparse tensor by indices
        """
        target = target.index_select(0, gram).coalesce().values()

        filter_indices = ones_tensor(
            torch.stack(
                [
                    torch.arange(context.size(0), dtype=torch.long).to(target.device),
                    context,
                ],
                0,
            ),
            size=target.shape,
        )

        return (target * filter_indices).sum(1).to_dense().reshape(context_shape)

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
        gram = gram.repeat_interleave(context.size(1), 1).view(-1)
        context = context.view(-1)

        gt = self.index_sparse_tensor(
            self.co_occurrence_counts,
            gram,
            context,
            pred.shape,
        )
        w = self.index_sparse_tensor(
            self.weighted_prob,
            gram,
            context,
            pred.shape,
        )

        y = self.loss(pred, gt.log1p())
        return (y * w).mean()
