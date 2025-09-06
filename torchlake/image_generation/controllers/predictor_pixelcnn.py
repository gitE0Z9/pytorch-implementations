from itertools import product

import torch
from torch import nn
from tqdm import tqdm


class PixelCNNPredictor:
    def __init__(self, device: torch.device):
        self.device = device

    def run(
        self,
        model: nn.Module,
        output_shape: tuple[int, int, int, int],
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, C, H, W = output_shape

        model.eval()
        with torch.no_grad():
            placeholder = torch.zeros(*output_shape).to(self.device)
            for i, j, c in tqdm(product(range(H), range(W), range(C))):
                if cond is None:
                    p = model(placeholder[:, :, : (i + 1), :])
                else:
                    p = model(placeholder[:, :, : (i + 1), :], cond=cond)
                p = p[:, c, :, i, j].softmax(-1)
                placeholder[:, c, i, j] = (
                    torch.multinomial(p, 1).squeeze_(-1).float() / 255.0
                )

        return placeholder


class DiagonalLSTMPredictor:
    def __init__(self, device: torch.device):
        self.device = device

    def run(
        self,
        model: nn.Module,
        output_shape: tuple[int, int, int, int],
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, C, H, W = output_shape

        model.eval()
        with torch.no_grad():
            placeholder = torch.zeros(*output_shape).to(self.device)
            for i, j, c in tqdm(product(range(H), range(W), range(C))):
                if cond is None:
                    p = model(placeholder)
                else:
                    p = model(placeholder, cond=cond)
                p = p[:, c, :, i, j].softmax(-1)
                placeholder[:, c, i, j] = (
                    torch.multinomial(p, 1).squeeze_(-1).float() / 255.0
                )

        return placeholder
