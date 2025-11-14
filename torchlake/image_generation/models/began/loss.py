from typing import Literal
import torch
from torch import nn
import torch.nn.functional as F


class BEGANDiscriminatorLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 1,
        k: float = 0,
        lambda_k: float = 1e-3,
        reduction: Literal["sum", "mean"] | None = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.k = k
        self.lambda_k = lambda_k
        self._k = k
        self.reduction = reduction

    def reset_k(self):
        self._k = self.k

    def update_k(self, loss_x: torch.Tensor, loss_xhat: torch.Tensor):
        self._k += self.lambda_k * (self.gamma * loss_x - loss_xhat)

    def forward(
        self,
        yhat_x: torch.Tensor,
        yhat_xhat: torch.Tensor,
        x: torch.Tensor,
        xhat: torch.Tensor,
    ) -> torch.Tensor:
        loss_x = F.l1_loss(yhat_x, x, reduction=self.reduction)
        loss_xhat = F.l1_loss(yhat_xhat, xhat, reduction=self.reduction)

        with torch.no_grad():
            self.update_k(loss_x, loss_xhat)

        return loss_x + self._k * loss_xhat
