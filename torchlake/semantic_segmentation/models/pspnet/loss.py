import torch
import torch.nn.functional as F
from torch import nn


class PSPLoss(nn.Module):

    def __init__(self, auxiliary_weight: float = 0.4):
        super().__init__()
        self.auxiliary_weight = auxiliary_weight

    def forward(
        self,
        pred: torch.Tensor,
        aux: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """prediction head classification loss and shallower layer classification loss

        Args:
            pred (torch.Tensor): prediction
            aux (torch.Tensor): auxiliary feature map
            target (torch.Tensor): target

        Returns:
            torch.Tensor: loss
        """
        return F.cross_entropy(pred, target) + self.auxiliary_weight * F.cross_entropy(
            aux, target
        )
