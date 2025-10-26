from typing import Literal
from torch import nn
import torch
import torch.nn.functional as F
from ..gan.loss import GANDiscriminatorLoss, GANGeneratorLoss


class ACGANDiscriminatorLoss(GANDiscriminatorLoss):
    def forward(
        self,
        yhat_x: torch.Tensor,
        yhat_xhat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return (
            super().forward(yhat_x[:, :1], yhat_xhat[:, :1])
            + F.cross_entropy(
                yhat_x[:, 1:],
                y,
                reduction=self.reduction,
            )
            + F.cross_entropy(
                yhat_xhat[:, 1:],
                y,
                reduction=self.reduction,
            )
        )


class ACGANGeneratorLoss(GANGeneratorLoss):
    def forward(
        self,
        yhat_xhat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(yhat_xhat[:, :1]) + F.cross_entropy(
            yhat_xhat[:, 1:],
            y,
            reduction=self.reduction,
        )
