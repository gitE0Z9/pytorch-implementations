from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class VAELoss(nn.Module):
    def __init__(
        self,
        kld_weight: int = 1,
        loss_type: Literal["mse", "bce"] = "mse",
        reduction: Literal["sum", "mean"] | None = "sum",
    ):
        super().__init__()
        self.kld_weight = kld_weight
        self.loss_func = {
            "mse": F.mse_loss,
            "bce": F.binary_cross_entropy_with_logits,
        }[loss_type]
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction_loss = self.loss_func(pred, gt, reduction=self.reduction)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return reconstruction_loss + self.kld_weight * kld_loss
