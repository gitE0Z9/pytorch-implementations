from typing import Iterable

import torch
from torch import nn

from .trainer_gan import GANTrainer


class CGANTrainer(GANTrainer):
    def train_discriminator(
        self,
        row: tuple[Iterable],
        noise: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, _, c = row
        x, c = x.to(self.device), c.to(self.device)

        with torch.no_grad():
            xhat = generator(torch.cat((noise, c), 1))

        if len(c.shape) == 2:
            c = c[:, :, None, None].expand(*c.shape, *xhat.shape[2:])

        return criterion(
            discriminator(torch.cat((x, c), 1)),
            discriminator(torch.cat((xhat, c), 1)),
        )

    def train_generator(
        self,
        row: tuple[Iterable],
        noise: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, _, c = row
        c = c.to(self.device)

        xhat = generator(torch.cat((noise, c), 1))

        if len(c.shape) == 2:
            c = c[:, :, None, None].expand(*c.shape, *xhat.shape[2:])

        return criterion(discriminator(torch.cat((xhat, c), 1)))
