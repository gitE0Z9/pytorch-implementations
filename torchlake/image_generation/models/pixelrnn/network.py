from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from ..pixelcnn.network import MaskedConv2d, split_mask_groups


class RowLSTM(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        kernel: int,
        mask_groups: int = 1,
    ):
        """Row LSTM, a form of PixelRNN learned row by row

        Args:
            input_channel (int): input channel size
            hidden_dim (int): dimension of hidden layers
            kernel (int): kernel size of the input-to-state and the state-to-state layers
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_groups = mask_groups

        # is: input to state
        self.conv_is = MaskedConv2d(
            input_channel,
            4 * hidden_dim,
            kernel=(1, kernel),
            mask_type="B",
            padding=(0, kernel // 2),
            mask_groups=mask_groups,
        )
        # ss: state to state
        self.conv_ss = nn.Conv2d(
            hidden_dim,
            4 * hidden_dim,
            kernel_size=(1, kernel),
            padding=(0, kernel // 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, _, H, W = x.size()
        h = (
            h
            if h is not None
            else torch.zeros(B, self.hidden_dim, 1, W, device=x.device)
        )
        c = (
            c
            if c is not None
            else torch.zeros(B, self.hidden_dim, 1, W, device=x.device)
        )

        z_is = self.conv_is(x)
        if cond is not None:
            z_is = z_is + cond

        outputs = []
        for i in range(H):
            # use conv to get row level representation
            z = z_is[:, :, i : i + 1, :] + self.conv_ss(h)
            sigmoid_states, tanh_state = split_mask_groups(
                z,
                self.mask_groups,
                (
                    3 * self.hidden_dim // self.mask_groups,
                    self.hidden_dim // self.mask_groups,
                ),
            )
            i, f, o = sigmoid_states.sigmoid().chunk(3, 1)
            g = tanh_state.tanh()
            c = f * c + i * g
            h = o * c.tanh()
            outputs.append(h)
        return torch.cat(outputs, dim=2)


class DiagonalLSTMCell(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        kernel: int,
        mask_groups: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel = kernel
        self.mask_groups = mask_groups

        # is: input to state
        self.conv_is = MaskedConv2d(
            input_channel,
            4 * hidden_dim,
            kernel=1,
            mask_type="B",
            mask_groups=mask_groups,
        )
        # ss: state to state
        self.conv_ss = nn.Conv2d(
            hidden_dim,
            4 * hidden_dim,
            kernel_size=(kernel, 1),
        )

    @staticmethod
    def skew(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        y = torch.zeros(B, C, H, 2 * W - 1).to(x.device)

        row_indices = torch.arange(H, device=x.device).view(-1, 1)
        col_indices = torch.arange(W, device=x.device).view(1, -1) + row_indices
        y[:, :, row_indices, col_indices] = x

        return y

    @staticmethod
    def unskew(x: torch.Tensor) -> torch.Tensor:
        _, _, H, W_PRIME = x.size()
        W = (W_PRIME + 1) // 2

        row_indices = torch.arange(H, device=x.device).view(-1, 1)
        col_indices = torch.arange(W, device=x.device).view(1, -1) + row_indices

        return x[:, :, row_indices, col_indices]

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, _, _, W = x.size()

        z_is = self.conv_is(x)
        if cond is not None:
            z_is = z_is + cond
        z_is = self.skew(z_is)

        outputs = []
        for j in range(2 * W - 1):
            # diagonal
            z = z_is[:, :, :, j : j + 1] + self.conv_ss(
                # pad size is backward, so w first then h, and so on.
                F.pad(h, (0, 0, self.kernel - 1, 0))
            )
            # lstm
            sigmoid_states, tanh_state = split_mask_groups(
                z,
                self.mask_groups,
                (
                    3 * self.hidden_dim // self.mask_groups,
                    self.hidden_dim // self.mask_groups,
                ),
            )
            i, f, o = sigmoid_states.sigmoid().chunk(3, 1)
            g = tanh_state.tanh()
            c = f * c + i * g
            h = o * c.tanh()
            outputs.append(h)
        outputs = torch.cat(outputs, dim=3)
        outputs = self.unskew(outputs)

        return outputs


class DiagonalLSTM(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        kernel: int,
        bidirectional: bool = False,
        mask_groups: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel = kernel
        self.bidirectional = bidirectional
        self.factor = 2 if self.bidirectional else 1
        self.mask_groups = mask_groups

        self.lstm_l = DiagonalLSTMCell(
            input_channel,
            hidden_dim,
            kernel,
            mask_groups=mask_groups,
        )
        if bidirectional:
            self.lstm_r = DiagonalLSTMCell(
                input_channel,
                hidden_dim,
                kernel,
                mask_groups=mask_groups,
            )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, _, H, _ = x.size()
        h = (
            h
            if h is not None
            else torch.zeros(self.factor, B, self.hidden_dim, H, 1, device=x.device)
        )
        c = (
            c
            if c is not None
            else torch.zeros(self.factor, B, self.hidden_dim, H, 1, device=x.device)
        )

        zl = self.lstm_l(x, h[0], c[0], cond)
        if self.bidirectional:
            zr = self.lstm_r(x.flip(3), h[1], c[1], cond)
            # move one row down, so the future information will not flow back
            zl[:, :, 1:, :] = zl[:, :, 1:, :] + zr[:, :, :-1, :]

        return zl


class BottleNeck(nn.Sequential):
    def __init__(
        self,
        hidden_dim: int,
        kernel: int,
        type: Literal["row", "diag"],
        bidirectional: bool = False,
        mask_groups: int = 1,
    ):
        super().__init__(
            (
                RowLSTM(
                    2 * hidden_dim,
                    hidden_dim,
                    kernel,
                    mask_groups=mask_groups,
                )
                if type == "row"
                else DiagonalLSTM(
                    2 * hidden_dim,
                    hidden_dim,
                    kernel,
                    bidirectional=bidirectional,
                    mask_groups=mask_groups,
                )
            ),
            MaskedConv2d(
                hidden_dim,
                2 * hidden_dim,
                1,
                mask_type="B",
                mask_groups=mask_groups,
            ),
        )
