import torch
import torch.nn.functional as F
from torch import nn


class HardDistillation(nn.Module):

    def __init__(self, alpha: float = 0.5, smooth: float = 0):
        super().__init__()
        assert 0 <= alpha <= 1, "alpha should fall in [0, 1]"

        self.alpha = alpha
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        pred_for_t: torch.Tensor,
        pred_of_t: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        s_loss = F.cross_entropy(pred, gt, label_smoothing=self.smooth)
        t_loss = F.cross_entropy(
            pred_for_t,
            pred_of_t.argmax(1),
            label_smoothing=self.smooth,
        )
        return (1 - self.alpha) * s_loss + self.alpha * t_loss
