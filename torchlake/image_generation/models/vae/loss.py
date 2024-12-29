from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class VAELoss(nn.Module):
    def __init__(
        self,
        kld_weight: int = 1,
        loss_type: Literal["mse", "bce"] = "mse",
    ):
        super().__init__()
        self.kld_weight = kld_weight
        self.loss_func = {
            "mse": F.mse_loss,
            "bce": F.binary_cross_entropy,
        }[loss_type]

    def forward(
        self,
        pred: torch.Tensor,
        mu: torch.Tensor,
        logsigma: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction_loss = self.loss_func(pred, gt, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())

        return reconstruction_loss + self.kld_weight * kld_loss
