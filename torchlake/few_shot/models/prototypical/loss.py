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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): shape (q, n, n). q is query size, n is n way

        Returns:
            torch.Tensor: negative log softmax.
        """
        q, n, _ = x.shape
        y = torch.arange(n).expand(q, n).to(x.device)
        logit = F.cross_entropy(-x, y)

        if self.reduction == "sum":
            return logit.sum()
        elif self.reduction == "mean":
            return logit.mean()
        else:
            return logit
