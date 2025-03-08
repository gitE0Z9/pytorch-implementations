import torch
import torch.nn.functional as F
from torch import nn


class DoublyStochasticAttentionLoss(nn.Module):

    def __init__(
        self,
        lambda_coef: float = 1e-5,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ):
        super().__init__()
        self.lambda_coef = lambda_coef
        self.cc = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        att: torch.Tensor,
    ) -> torch.Tensor:
        att_sum = att.sum(2)
        return self.cc(pred, gt) + self.lambda_coef * F.mse_loss(
            att_sum,
            torch.ones_like(att_sum),
            reduction="sum",
        )
