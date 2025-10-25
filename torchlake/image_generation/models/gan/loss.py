from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class GANDiscriminatorLoss(nn.Module):

    def __init__(self, reduction: Literal["sum", "mean"] | None = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, yhat_x: torch.Tensor, yhat_xhat: torch.Tensor) -> torch.Tensor:
        valid = torch.ones_like(yhat_x)

        real_loss = F.binary_cross_entropy_with_logits(
            yhat_x,
            valid,
            reduction=self.reduction,
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            yhat_xhat,
            1 - valid,
            reduction=self.reduction,
        )
        return real_loss + fake_loss


class GANGeneratorLoss(nn.Module):

    def __init__(self, reduction: Literal["sum", "mean"] | None = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, yhat_xhat: torch.Tensor) -> torch.Tensor:
        valid = torch.ones_like(yhat_xhat)

        return F.binary_cross_entropy_with_logits(
            yhat_xhat,
            valid,
            reduction=self.reduction,
        )
