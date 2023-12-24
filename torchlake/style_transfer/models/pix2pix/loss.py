import torch
import torch.nn.functional as F
from torch import nn


class Pix2PixGeneratorLoss(nn.Module):
    def __init__(self, lambda_coef: float):
        super(Pix2PixGeneratorLoss, self).__init__()
        self.lambda_coef = lambda_coef

    def forward(
        self,
        fake_discriminated: torch.Tensor,
        generated: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        g_loss = F.binary_cross_entropy_with_logits(
            fake_discriminated,
            torch.ones_like(fake_discriminated),
        )
        g2_loss = F.l1_loss(generated, label)
        loss_G = g_loss + self.lambda_coef * g2_loss

        return loss_G


class Pix2PixDiscriminatorLoss(nn.Module):
    def __init__(self):
        super(Pix2PixDiscriminatorLoss, self).__init__()

    def forward(
        self,
        real_discriminated: torch.Tensor,
        fake_discriminated: torch.Tensor,
    ) -> torch.Tensor:
        real_loss = F.binary_cross_entropy_with_logits(
            real_discriminated,
            torch.ones_like(real_discriminated),
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_discriminated,
            torch.zeros_like(fake_discriminated),
        )
        loss_D = (real_loss + fake_loss) / 2

        return loss_D
