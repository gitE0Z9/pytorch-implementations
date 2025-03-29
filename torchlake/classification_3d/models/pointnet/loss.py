import torch
import torch.nn.functional as F
from torch import nn


class PointNetLoss(nn.Module):

    def __init__(self, coef_orthonormal: float = 1e-3):
        super().__init__()
        self.coef_orthonormal = coef_orthonormal

    def forward(
        self,
        pred: torch.Tensor,
        transforms: tuple[torch.Tensor],
        gt: torch.Tensor,
    ) -> torch.Tensor:
        main_loss = F.cross_entropy(pred, gt)
        reg_loss = sum(
            F.mse_loss(
                torch.bmm(transform.transpose(-1, -2), transform),
                torch.eye(transform.size(-1)).expand_as(transform).to(transform.device),
            )
            for transform in transforms
        )

        return main_loss + self.coef_orthonormal * reg_loss
