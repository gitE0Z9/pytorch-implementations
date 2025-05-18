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
        mask_groups: int = 1,
        **kwargs,
    ):
        super().__init__(input_channel, output_channel, kernel, **kwargs)
        self.mask_type = mask_type
        self.mask_groups = mask_groups
        self.register_buffer("mask", torch.zeros(self.weight.shape), persistent=False)
        self._build_mask()

        self.weight.data *= self.mask
        self.weight.register_hook(self._mask_gradients)

    def _build_mask(self):
        k = self.kernel_size[0]
        center = k // 2
        # up
        self.mask[:, :, :center, :] = 1

        h = self.in_channels // self.mask_groups
        if self.mask_type == "A":
            # left
            self.mask[:, :, center, :center] = 1
            # present channel dependency
            for out_g in range(1, self.mask_groups):
                for in_g in range(self.mask_groups - 1):
                    if out_g == in_g:
                        continue
                    self.mask[
                        h * out_g : h * (out_g + 1),
                        h * in_g : h * (in_g + 1),
                        center,
                        center,
                    ] = 1
        else:  # mask_type == 'B'
            # left include present pixel
            self.mask[:, :, center, : center + 1] = 1
            # present channel dependency
            for out_g in range(self.mask_groups):
                for in_g in range(self.mask_groups):
                    if out_g >= in_g:
                        continue
                    self.mask[
                        h * out_g : h * (out_g + 1),
                        h * in_g : h * (in_g + 1),
                        center,
                        center,
                    ] = 0

    def _mask_gradients(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self.mask


class BottleNeck(nn.Sequential):
    def __init__(self, hidden_dim: int):
        super().__init__(
            nn.ReLU(),
            MaskedConv2d(2 * hidden_dim, hidden_dim, 1, mask_type="B"),
            nn.ReLU(),
            MaskedConv2d(hidden_dim, hidden_dim, 3, mask_type="B", padding=1),
            nn.ReLU(),
            MaskedConv2d(hidden_dim, 2 * hidden_dim, 1, mask_type="B"),
        )


class GatedLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel: int,
    ):
        super().__init__()
        self.cf = nn.Conv2d(2 * hidden_dim, hidden_dim, kernel, padding=kernel // 2)
        self.cg = nn.Conv2d(2 * hidden_dim, hidden_dim, kernel, padding=kernel // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.split(2, 1)
        return self.cf(x1).tanh() * self.cg(x2).sigmoid()
