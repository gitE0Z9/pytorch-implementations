from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class PrototypicalNetLoss(nn.Module):
    def __init__(
        self,
        reduction: Literal["sum", "mean"] | None = "mean",
    ):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): shape (q, n, n). q is query size, n is n way
            y (torch.Tensor): shape (q, n, n). q is query size, n is n way

        Returns:
            torch.Tensor: negative log softmax.
        """
        logit = F.cross_entropy(-x, y)

        if self.reduction == "sum":
            return logit.sum()
        elif self.reduction == "mean":
            return logit.mean()
        else:
            return logit
