from typing import Generator

import torch
from torch import nn


class CGANPredictor:
    def __init__(self, device: torch.device):
        self.device = device

    def run(self, noise_generator: Generator, cond: torch.Tensor, model: nn.Module):
        model.eval()
        with torch.no_grad():
            z = next(noise_generator).to(self.device)
            z = torch.cat((z, cond), 1)
            return (model(z) + 1) / 2
