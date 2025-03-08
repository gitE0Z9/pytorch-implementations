import torch
import torch.nn.functional as F
from torch import nn


class DoublyStochasticAttentionLoss(nn.Module):

    def __init__(self, lambda_coef: float = 1e-5):
        super().__init__()
        self.lambda_coef = lambda_coef

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        att: torch.Tensor,
    ) -> torch.Tensor:
        att_sum = att.sum(1)
        return F.cross_entropy(pred, gt) + self.lambda_coef * F.mse_loss(
            att_sum, torch.ones_like(att_sum)
        )
