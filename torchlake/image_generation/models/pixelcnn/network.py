from typing import Literal

import torch
from torch import nn


class MaskedConv2d(nn.Conv2d):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        mask_type: Literal["A", "B"] = "A",
        **kwargs,
    ):
        super().__init__(input_channel, output_channel, kernel, **kwargs)
        self.mask_type = mask_type
        self.register_buffer("mask", torch.zeros(self.weight.shape), persistent=False)
        self._build_mask()

        self.weight.data *= self.mask
        self.weight.register_hook(self._mask_gradients)

    def _build_mask(self):
        k = self.kernel_size[0]
        center = k // 2
        self.mask[:, :, :center, :] = 1
        if self.mask_type == "A":
            self.mask[:, :, center, :center] = 1
        else:  # mask_type == 'B'
            self.mask[:, :, center, : center + 1] = 1

    def _mask_gradients(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self.mask


class BottleNeck(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(2 * hidden_dim, hidden_dim, 1, mask_type="B"),
            nn.ReLU(),
            MaskedConv2d(hidden_dim, hidden_dim, 3, mask_type="B", padding=1),
            nn.ReLU(),
            MaskedConv2d(hidden_dim, 2 * hidden_dim, 1, mask_type="B"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
