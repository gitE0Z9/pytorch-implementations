from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class LSGANDiscriminatorLoss(nn.Module):

    def __init__(
        self,
        encoding: Literal["binary", "triple"] = "binary",
        reduction: Literal["sum", "mean"] | None = "mean",
    ):
        super().__init__()
        self.encoding = encoding
        self.reduction = reduction

    def forward(self, yhat_x: torch.Tensor, yhat_xhat: torch.Tensor) -> torch.Tensor:
        valid = torch.ones_like(yhat_x)
        invalid = 1 - valid if self.encoding == "binary" else -valid

        real_loss = F.mse_loss(yhat_x, valid, reduction=self.reduction)
        fake_loss = F.mse_loss(yhat_xhat, invalid, reduction=self.reduction)
        return real_loss + fake_loss


class LSGANGeneratorLoss(nn.Module):

    def __init__(
        self,
        encoding: Literal["binary", "triple"] = "binary",
        reduction: Literal["sum", "mean"] | None = "mean",
    ):
        super().__init__()
        self.encoding = encoding
        self.reduction = reduction

    def forward(self, yhat_xhat: torch.Tensor) -> torch.Tensor:
        valid = (
            torch.ones_like(yhat_xhat)
            if self.encoding == "binary"
            else torch.zeros_like(yhat_xhat)
        )

        return F.mse_loss(
            yhat_xhat,
            valid,
            reduction=self.reduction,
        )
