from typing import Iterable

import torch
from torch import nn

from .trainer_gan import GANTrainer


class EBGANTrainer(GANTrainer):
    def train_generator(
        self,
        row: tuple[Iterable],
        noise: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
    ) -> torch.Tensor:
        xhat = generator(noise)

        output_latent = criterion.lambda_pt > 0
        yhat = discriminator(xhat, output_latent=output_latent)
        if output_latent:
            yhat, z = yhat
            return criterion(yhat, xhat, z)

        return criterion(yhat, xhat)

    def train_discriminator(
        self,
        row: tuple[Iterable],
        noise: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, _ = row
        x = x.to(self.device)

        with torch.no_grad():
            xhat = generator(noise)

        return criterion(discriminator(x), discriminator(xhat), x, xhat)
