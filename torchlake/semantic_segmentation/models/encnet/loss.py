import torch
import torch.nn.functional as F
from torch import nn


class EncNetLoss(nn.Module):
    def __init__(
        self,
        se_weight: float = 0.2,
        auxiliary_weight: float = 0.4,
        return_all_loss: bool = False,
    ):
        super().__init__()
        self.se_weight = se_weight
        self.auxiliary_weight = auxiliary_weight
        self.return_all_loss = return_all_loss

    def forward(
        self,
        x: torch.Tensor,
        se: torch.Tensor,
        aux: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.cross_entropy(x, gt)

        if self.se_weight > 0:
            presences = torch.zeros(
                x.size(0), x.size(1), device=gt.device, dtype=torch.float
            )
            presences.scatter_(1, gt.view(x.size(0), -1), 1)
            se_loss = self.se_weight * F.binary_cross_entropy_with_logits(se, presences)
            loss = loss + se_loss

        if self.auxiliary_weight > 0:
            aux_loss = self.auxiliary_weight * F.cross_entropy(aux, gt)
            loss = loss + aux_loss

        if self.return_all_loss:
            return loss, se_loss, aux_loss

        return loss
