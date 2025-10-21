from typing import Literal, Sequence

import torch
from torch import nn


def split_mask_groups(
    x: torch.Tensor,
    mask_groups: int,
    splits: Sequence[int],
) -> list[torch.Tensor]:
    B, _, H, W = x.shape
    x = x.view(B, mask_groups, -1, H, W)

    output = []
    offset = 0
    for split in splits:
        output.append(x[:, :, offset : offset + split].reshape(B, -1, H, W))
        offset += split

    return output


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
        k_y, k_x = self.kernel_size
        center_y, center_x = k_y // 2, k_x // 2
        # up
        self.mask[:, :, :center_y, :] = 1

        i_group_channels = self.in_channels // self.mask_groups
        o_group_channels = self.out_channels // self.mask_groups
        if self.mask_type == "A":
            # left
            self.mask[:, :, center_y, :center_x] = 1
            # present channel dependency
            for out_g in range(1, self.mask_groups):
                for in_g in range(self.mask_groups - 1):
                    if out_g == in_g:
                        continue
                    self.mask[
                        o_group_channels * out_g : o_group_channels * (out_g + 1),
                        i_group_channels * in_g : i_group_channels * (in_g + 1),
                        center_y,
                        center_x,
                    ] = 1
        else:  # mask_type == 'B'
            # left include present pixel
            self.mask[:, :, center_y, : center_x + 1] = 1
            # present channel dependency
            for out_g in range(self.mask_groups):
                for in_g in range(self.mask_groups):
                    if out_g >= in_g:
                        continue
                    self.mask[
                        o_group_channels * out_g : o_group_channels * (out_g + 1),
                        i_group_channels * in_g : i_group_channels * (in_g + 1),
                        center_y,
                        center_x,
                    ] = 0

    def _mask_gradients(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self.mask


class BottleNeck(nn.Sequential):
    def __init__(
        self,
        hidden_dim: int,
        mask_groups: int = 1,
    ):
        super().__init__(
            nn.ReLU(),
            MaskedConv2d(
                2 * hidden_dim,
                hidden_dim,
                1,
                mask_type="B",
                mask_groups=mask_groups,
            ),
            nn.ReLU(),
            MaskedConv2d(
                hidden_dim,
                hidden_dim,
                3,
                mask_type="B",
                padding=1,
                mask_groups=mask_groups,
            ),
            nn.ReLU(),
            MaskedConv2d(
                hidden_dim,
                2 * hidden_dim,
                1,
                mask_type="B",
                mask_groups=mask_groups,
            ),
        )
