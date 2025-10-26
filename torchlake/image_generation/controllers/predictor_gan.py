from typing import Generator

import torch
from torch import nn


class GANPredictor:
    def __init__(self, device: torch.device):
        self.device = device

    def run(self, noise_generator: Generator, model: nn.Module):
        model.eval()
        with torch.no_grad():
            z = next(noise_generator).to(self.device)
            return (model(z) + 1) / 2
