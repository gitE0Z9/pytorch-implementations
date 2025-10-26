from typing import Iterable, Literal

import torch
from torch import nn

from .trainer_cgan import CGANTrainer


class ACGANTrainer(CGANTrainer):

    def train_discriminator(
        self,
        row: tuple[Iterable],
        noise: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, y, c = row
        x, y, c = x.to(self.device), y.to(self.device), c.to(self.device)

        with torch.no_grad():
            xhat = generator(torch.cat((noise, c), 1))

        if len(c.shape) == 2:
            c = c[:, :, None, None].expand(*c.shape, *xhat.shape[2:])

        return criterion(discriminator(x), discriminator(xhat), y)

    def train_generator(
        self,
        row: tuple[Iterable],
        noise: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y, c = row
        y, c = y.to(self.device), c.to(self.device)

        xhat = generator(torch.cat((noise, c), 1))
        return criterion(discriminator(xhat), y)
