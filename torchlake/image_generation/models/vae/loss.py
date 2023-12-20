import torch
from torch import nn
import torch.nn.functional as F


class VaeLoss(nn.Module):
    def __init__(self, kld_weight: int = 1):
        super(VaeLoss, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, x: torch.Tensor, y: torch.Tensor, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        reconstruct_loss = F.mse_loss(x, y, reduction="sum")
        #         reconstruct_loss = F.binary_cross_entropy(output, img_flatten,reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
        loss = reconstruct_loss + self.kld_weight * kld_loss

        return loss
