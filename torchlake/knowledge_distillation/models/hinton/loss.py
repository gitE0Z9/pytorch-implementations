import torch
from torch import nn
import torch.nn.functional as F


class KLDLoss(nn.Module):

    def __init__(self, alpha: float = 0.5, temp: float = 1):
        super().__init__()
        assert 0 <= alpha <= 1, "alpha should fall in [0, 1]"
        self.alpha = alpha
        self.temp = temp

    def forward(
        self,
        pred: torch.Tensor,
        t_pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        s_loss = F.cross_entropy(pred, gt)
        t_loss = F.kl_div(
            F.log_softmax(pred / self.temp, dim=-1),
            t_pred / self.temp,
            reduction="batchmean",
        )
        return (1 - self.alpha) * s_loss + self.alpha * (self.temp**2) * t_loss
