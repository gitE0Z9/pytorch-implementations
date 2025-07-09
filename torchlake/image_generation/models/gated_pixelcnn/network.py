import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..pixelcnn.network import MaskedConv2d


class DownwardConv2d(nn.Conv2d):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel: int,
        **kwargs,
    ):
        # avoid conflict with kernel_size
        self._kernel = kernel
        self._kh = math.ceil(kernel / 2)
        super().__init__(
            input_channel,
            output_channel,
            (self._kh, kernel),
            padding=(0, kernel // 2),
            **kwargs,
        )

    @staticmethod
    def shift_downward(x: torch.Tensor, offset: int):
        return F.pad(x, (0, 0, offset, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.shift_downward(x, self._kh - 1)
        y = super().forward(x)
        return self.shift_downward(y, 1)


class GatedLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel: int,
        conditional_shape: Sequence[int] | None = None,
        mask_groups: int = 1,
    ):
        """The layer of Gated PixelCNN, including both of vertical and horizontal branches.

        Args:
            hidden_dim (int): hidden dimension
            kernel (int): kernel size
            conditional_shape (Sequence[int] | None, optional): the shape of the conditional representation. Default is None.
        """
        self.hidden_dim = hidden_dim
        self.conditional_shape = conditional_shape
        super().__init__()

        # vertical branch
        self.conv_v = DownwardConv2d(
            hidden_dim,
            2 * hidden_dim,
            kernel,
        )
        self.conv_v_gate_f = DownwardConv2d(
            hidden_dim,
            hidden_dim,
            kernel,
        )
        self.conv_v_gate_g = DownwardConv2d(
            hidden_dim,
            hidden_dim,
            kernel,
        )

        # neck
        self.conv_v_h = nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, 1)

        # horizontal branch
        self.conv_h = MaskedConv2d(
            hidden_dim,
            2 * hidden_dim,
            kernel=(1, kernel),
            padding=(0, kernel // 2),
            mask_type="B",
            mask_groups=mask_groups,
        )
        self.conv_h_gate_f = MaskedConv2d(
            hidden_dim,
            hidden_dim,
            kernel,
            padding=kernel // 2,
            mask_type="B",
            mask_groups=mask_groups,
        )
        self.conv_h_gate_g = MaskedConv2d(
            hidden_dim,
            hidden_dim,
            kernel,
            padding=kernel // 2,
            mask_type="B",
            mask_groups=mask_groups,
        )
        self.conv_h_output = MaskedConv2d(
            hidden_dim,
            hidden_dim,
            1,
            mask_type="B",
            mask_groups=mask_groups,
        )

        if conditional_shape is not None:
            conditional_input_channel = conditional_shape[0]
            self.cond_h_f = nn.Conv2d(conditional_input_channel, hidden_dim, 1)
            self.cond_h_g = nn.Conv2d(conditional_input_channel, hidden_dim, 1)
            self.cond_v_f = nn.Conv2d(conditional_input_channel, hidden_dim, 1)
            self.cond_v_g = nn.Conv2d(conditional_input_channel, hidden_dim, 1)

    def forward(
        self,
        v: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """forward

        Args:
            v (torch.Tensor): vertical tensor
            h (torch.Tensor): horizontal tensor.
            c (torch.Tensor | None, optional): conditional tensor. Defaults to None.

        Returns:
            torch.Tensor: feature map
        """
        # z means latent representation
        # v means vertical, h means horizontal
        zv = self.conv_v(v)
        zh = self.conv_h(h) + self.conv_v_h(zv)

        vf = self.conv_v_gate_f(zv[:, : self.hidden_dim, :, :])
        vg = self.conv_v_gate_g(zv[:, -self.hidden_dim :, :, :])
        hf = self.conv_h_gate_f(zh[:, : self.hidden_dim, :, :])
        hg = self.conv_h_gate_g(zh[:, -self.hidden_dim :, :, :])

        if c is not None and self.conditional_shape is not None:
            vf = vf + self.cond_v_f(c)
            vg = vg + self.cond_v_g(c)
            hf = hf + self.cond_h_f(c)
            hg = hg + self.cond_h_g(c)

        zv = vf.tanh() * vg.sigmoid()

        zh = self.conv_h_output(hf.tanh() * hg.sigmoid())
        zh = zh + h

        return zv, zh
