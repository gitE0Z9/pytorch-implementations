from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class EBGANDiscriminatorLoss(nn.Module):

    def __init__(
        self,
        m: float = 10,
        reduction: Literal["sum", "mean"] | None = "mean",
    ):
        super().__init__()
        self.m = m
        self.reduction = reduction

    def forward(
        self,
        yhat_x: torch.Tensor,
        yhat_xhat: torch.Tensor,
        x: torch.Tensor,
        xhat: torch.Tensor,
    ) -> torch.Tensor:
        x_mse = F.mse_loss(yhat_x, x, reduction="none")
        xhat_mse = F.mse_loss(yhat_xhat, xhat, reduction="none")
        return F.hinge_embedding_loss(
            torch.cat((x_mse, xhat_mse)),
            torch.cat((torch.ones_like(x_mse), -torch.ones_like(xhat_mse))),
            margin=self.m,
            reduction=self.reduction,
        )


class EBGANGeneratorLoss(nn.Module):

    def __init__(
        self,
        lambda_pt: float = 0.1,
        reduction: Literal["sum", "mean"] | None = "mean",
    ):
        super().__init__()
        self.lambda_pt = lambda_pt
        self.reduction = reduction

    def forward(
        self,
        yhat_xhat: torch.Tensor,
        xhat: torch.Tensor,
        z_xhat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = F.mse_loss(yhat_xhat, xhat, reduction=self.reduction)

        if self.lambda_pt > 0:
            assert (
                z_xhat is not None
            ), "with pulling away term, latent vector must be provided"
            b, h = z_xhat.shape
            # B, h
            z_xhat_normed = z_xhat / z_xhat.norm(p=2, dim=1, keepdim=True)
            # cosine similarity
            # B, h, h
            cos = torch.einsum("bi,bj->bij", z_xhat_normed, z_xhat_normed) ** 2
            i = torch.arange(h).expand(b, h)
            cos[:, i, i] = 0

            if self.reduction == "sum":
                return loss + self.lambda_pt * cos.sum()
            elif self.reduction == "mean":
                return loss + self.lambda_pt * cos.mean()
            else:
                return loss, cos

        return loss
