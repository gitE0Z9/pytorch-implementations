from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def pad_on_top(x: torch.Tensor, offset: int):
    return F.pad(x, (0, 0, offset, 0))


def pad_on_left(x: torch.Tensor, offset: int):
    return F.pad(x, (offset, 0, 0, 0))


def shift_downward(x: torch.Tensor, offset: int):
    return pad_on_top(x, offset)[:, :, :-offset, :]


def shift_rightward(x: torch.Tensor, offset: int):
    return pad_on_left(x, offset)[:, :, :, :-offset]


# TODO: weight norm
# TODO: logistic mixture sample


class ConcatELU(nn.Module):
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(torch.cat([x, -x], 1), alpha=self.alpha, inplace=self.inplace)


class DownwardConv2d(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kh, _ = self.kernel_size
        y = pad_on_top(x, kh - 1)
        return super().forward(y)


class DownwardAndRightwardConv2d(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kh, kw = self.kernel_size
        y = pad_on_top(x, kh - 1)
        y = pad_on_left(y, kw - 1)
        return super().forward(y)


class DownwardConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)

        kh, kw = self.kernel_size
        # problematic
        if kh == 1:
            return y
        return y[:, :, : -(kh - 1), kw // 2 : -(kw // 2)]


class DownwardAndRightwardConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)

        kh, kw = self.kernel_size
        # problematic
        if kh == 1 and kw != 1:
            return y[:, :, :, : -(kw - 1)]
        elif kh != 1 and kw == 1:
            return y[:, :, : -(kh - 1), :]
        else:
            return y[:, :, : -(kh - 1), : -(kw - 1)]


class DownsampleLayer(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        vertical_kernel: Sequence[int] = (2, 3),
        horizontal_kernel: Sequence[int] = (2, 2),
    ):
        super().__init__()
        # see all of above context and self
        self.v = DownwardConv2d(
            input_channel,
            output_channel,
            vertical_kernel,
            stride=2,
            padding=(0, vertical_kernel[1] // 2),
        )
        # look left top include self
        self.h = DownwardAndRightwardConv2d(
            input_channel,
            output_channel,
            horizontal_kernel,
            stride=2,
        )

    def forward(
        self,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.v(v), self.h(h)


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        vertical_kernel: Sequence[int] = (2, 3),
        horizontal_kernel: Sequence[int] = (2, 2),
    ):
        super().__init__()
        # target shape is I * S + K - 1
        # O = (I-1)*S + K - 2P + OP, padding="VALID"
        self.v = DownwardConvTranspose2d(
            input_channel,
            output_channel,
            vertical_kernel,
            stride=2,
            output_padding=1,
        )
        self.h = DownwardAndRightwardConvTranspose2d(
            input_channel,
            output_channel,
            horizontal_kernel,
            stride=2,
            output_padding=1,
        )

    def forward(
        self,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.v(v), self.h(h)


class ResidualLayer(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        vertical_kernel: Sequence[int] = (2, 3),
        horizontal_kernel: Sequence[int] = (2, 2),
        is_upside: bool = False,
        dropout_prob: float = 0.1,
        conditional_shape: Sequence[int] | None = None,
    ):
        """The layer of PixelCNN++, including both of vertical and horizontal branches.

        Args:
            input_channel (int): input channel
            hidden_dim (int): hidden dimension
            vertical_kernel (int | Sequence[int], optional): kernel size. Defaults to (2, 3).
            horizontal_kernel (int | Sequence[int], optional): kernel size. Defaults to (2, 2).
            is_upside (bool, optional): is upsampling side of UNet. Defaults to False.
            dropout_prob (float, optional): dropout probability. Defaults to 0.1.
            conditional_shape (Sequence[int] | None, optional): the shape of the conditional representation. Default is None.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.is_upside = is_upside
        self.conditional_shape = conditional_shape

        self.layers_v = nn.Sequential(
            ConcatELU(),
            DownwardConv2d(
                2 * input_channel,
                hidden_dim,
                vertical_kernel,
                padding=(0, vertical_kernel[1] // 2),
            ),
        )
        self.layers_v[-1] = torch.nn.utils.weight_norm(self.layers_v[-1])

        self.gate_v = nn.Sequential(
            ConcatELU(),
            nn.Dropout(p=dropout_prob),
            DownwardConv2d(
                2 * hidden_dim,
                2 * hidden_dim,
                vertical_kernel,
                padding=(0, vertical_kernel[1] // 2),
            ),
        )
        self.gate_v[-1] = torch.nn.utils.weight_norm(self.gate_v[-1])

        self.layers_h = nn.Sequential(
            ConcatELU(),
            DownwardAndRightwardConv2d(
                2 * input_channel,
                hidden_dim,
                horizontal_kernel,
            ),
        )
        self.layers_h[-1] = torch.nn.utils.weight_norm(self.layers_h[-1])

        self.gate_h = nn.Sequential(
            ConcatELU(),
            nn.Dropout(p=dropout_prob),
            DownwardAndRightwardConv2d(
                2 * hidden_dim,
                2 * hidden_dim,
                horizontal_kernel,
            ),
        )
        self.gate_h[-1] = torch.nn.utils.weight_norm(self.gate_h[-1])

        if self.is_upside:
            self.skip_v = nn.Sequential(
                ConcatELU(),
                nn.Conv2d(2 * hidden_dim, hidden_dim, 1),
            )
            self.skip_v[-1] = torch.nn.utils.weight_norm(self.skip_v[-1])

            self.skip_h = nn.Sequential(
                ConcatELU(),
                nn.Conv2d(4 * hidden_dim, hidden_dim, 1),
            )
            self.skip_h[-1] = torch.nn.utils.weight_norm(self.skip_h[-1])

        else:
            self.v_to_h = nn.Sequential(
                ConcatELU(),
                nn.Conv2d(2 * hidden_dim, hidden_dim, 1),
            )
            self.v_to_h[-1] = torch.nn.utils.weight_norm(self.v_to_h[-1])

        if conditional_shape is not None:
            conditional_input_channel = conditional_shape[0]
            self.cond_layer_v = nn.Conv2d(conditional_input_channel, 2 * hidden_dim, 1)
            self.cond_layer_h = nn.Conv2d(conditional_input_channel, 2 * hidden_dim, 1)

    def forward(
        self,
        v: torch.Tensor,
        h: torch.Tensor,
        av: torch.Tensor | None = None,
        ah: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """forward

        Args:
            v (torch.Tensor): vertical input tensor
            h (torch.Tensor): horizontal input tensor.
            cond (torch.Tensor | None, optional): conditional tensor. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: vertical feature map, horizontal feature map
        """
        if cond is not None:
            assert (
                self.conditional_shape is not None
            ), "to process conditional feature map, conditional layers is a must."

        zv = self.layers_v(v)
        if self.is_upside:
            zv = zv + self.skip_v(av)
        zv = self.gate_v(zv)
        if cond is not None:
            zv = zv + self.cond_layer_v(cond)
        zv = zv[:, : self.hidden_dim, :, :] * zv[:, self.hidden_dim :, :, :].sigmoid()
        zv = v + zv

        zh = self.layers_h(h)
        if self.is_upside:
            zh = zh + self.skip_h(torch.cat([zv, ah], 1))
        else:
            zh = zh + self.v_to_h(zv)
        zh = self.gate_h(zh)
        if cond is not None:
            zh = zh + self.cond_layer_h(cond)
        zh = zh[:, : self.hidden_dim, :, :] * zh[:, self.hidden_dim :, :, :].sigmoid()
        zh = h + zh

        return zv, zh
