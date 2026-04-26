import torch
import torch.nn.functional as F
from torch import nn


class Pix2PixGeneratorLoss(nn.Module):
    def __init__(self, lambda_coef: float):
        super().__init__()
        self.lambda_coef = lambda_coef

    def forward(
        self,
        yhat_xhat: torch.Tensor,
        xhat: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        adversarial_loss = F.binary_cross_entropy_with_logits(
            yhat_xhat, torch.ones_like(yhat_xhat)
        )
        reconstruction_loss = F.l1_loss(xhat, x)
        return adversarial_loss + self.lambda_coef * reconstruction_loss


class Pix2PixDiscriminatorLoss(nn.Module):
    def forward(self, yhat_xhat: torch.Tensor, yhat_x: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            torch.cat([yhat_xhat, yhat_x], 1),
            torch.cat([torch.zeros_like(yhat_xhat), torch.ones_like(yhat_x)], 1),
        )
