import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..pixelcnn.network import MaskedConv2d


def pad_on_top(x: torch.Tensor, offset: int):
    return F.pad(x, (0, 0, offset, 0))


def pad_on_left(x: torch.Tensor, offset: int):
    return F.pad(x, (offset, 0, 0, 0))


def shift_downward(x: torch.Tensor, offset: int):
    return pad_on_top(x, offset)[:, :, :-offset, :]


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
        self._k = math.ceil(kernel / 2)
        super().__init__()

        # vertical branch
        self.conv_v = nn.Conv2d(
            hidden_dim,
            2 * hidden_dim,
            (self._k, kernel),
            padding=(0, kernel // 2),
        )
        self.conv_v_gate = nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, 1, groups=2)

        # neck
        self.conv_v_h = nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, 1)

        # horizontal branch
        self.conv_h = MaskedConv2d(
            hidden_dim,
            2 * hidden_dim,
            kernel=(1, self._k),
            mask_type="B",
            mask_groups=mask_groups,
        )
        self.conv_h_gate = MaskedConv2d(
            2 * hidden_dim,
            2 * hidden_dim,
            1,
            mask_type="B",
            mask_groups=mask_groups,
            groups=2,
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
        zv = self.conv_v(pad_on_top(v, self._k - 1))
        zh = self.conv_h(pad_on_left(h, self._k - 1)) + self.conv_v_h(
            shift_downward(zv, 1)
        )

        zv = self.conv_v_gate(zv)
        vf, vg = zv[:, : self.hidden_dim, :, :], zv[:, self.hidden_dim :, :, :]
        zh = self.conv_h_gate(zh)
        hf, hg = zh[:, : self.hidden_dim, :, :], zh[:, self.hidden_dim :, :, :]

        if c is not None and self.conditional_shape is not None:
            vf = vf + self.cond_v_f(c)
            vg = vg + self.cond_v_g(c)
            hf = hf + self.cond_h_f(c)
            hg = hg + self.cond_h_g(c)

        zv = vf.tanh() * vg.sigmoid()

        zh = self.conv_h_output(hf.tanh() * hg.sigmoid())
        zh = zh + h

        return zv, zh
