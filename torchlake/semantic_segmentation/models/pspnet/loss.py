import torch
import torch.nn.functional as F
from torch import nn


class PspLoss(nn.Module):

    def __init__(self, auxiliary_weight: float = 0.4):
        super(PspLoss, self).__init__()
        self.auxiliary_weight = auxiliary_weight

    def forward(
        self,
        pred: torch.Tensor,
        aux: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """x is for psp output classification loss, aux is for shallower layer classification loss

        Args:
            pred (torch.Tensor): prediction
            aux (torch.Tensor): auxiliary feature map
            target (torch.Tensor): target

        Returns:
            torch.Tensor: _description_
        """
        return F.cross_entropy(pred, target) + self.auxiliary_weight * F.cross_entropy(
            aux, target
        )
