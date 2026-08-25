import torch
from torch import nn
import torch.nn.functional as F

from ..utils.numerical import generate_grid


class PositionEncoding1d(nn.Module):

    def __init__(
        self,
        seq_len: int | None = None,
        hidden_dim: int | None = None,
        is_fixed: bool = True,
        trainable: bool = False,
    ):
        """Position encoding for 1d sequence

        Args:
            seq_len (int | None, optional): sequence length for fixed size. Defaults to None.
            hidden_dim (int | None, optional): hidden dimension for fixed size. Defaults to None.
            is_fixed (bool, optional): is encoding parameters fixed size. Defaults to True.
            trainable (bool, optional): is encoding parameters fixed size and trainable parameters. Defaults to False.
        """
        if trainable:
            assert (
                is_fixed
            ), "trainable position encoding needs to be fixed into a predefined shape"
        if is_fixed:
            assert (
                seq_len is not None and hidden_dim is not None
            ), "must provide sequence length and hidden dimension to generate fixed encoding"

        super().__init__()
        self.trainable = trainable
        self.is_fixed = is_fixed

        if is_fixed:
            # 1, s, h
            theta = self.init_fourier_grid(seq_len, hidden_dim)
            if trainable:
                self.encoding = nn.Parameter(theta)
            else:
                self.register_buffer("encoding", theta)

    def init_fourier_grid(
        self,
        seq_len: int | None = None,
        hidden_dim: int | None = None,
    ) -> torch.Tensor:
        grid_x, grid_y = generate_grid(hidden_dim, seq_len)
        # TODO: if they truly share freq, just add pi/2 for sin to cos
        grid_x = grid_x[:, 0::2].repeat_interleave(2, 1)
        denominator = 10000 ** (2 * grid_x / hidden_dim)
        y = grid_y / denominator

        y[:, 0::2] = y[:, 0::2].sin()
        y[:, 1::2] = y[:, 1::2].cos()

        # 1, S, h
        return y.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """generate position encoding

        Args:
            x (torch.Tensor): input tensor, shape is (b, s, h)

        Returns:
            torch.Tensor: if trainable interpolated fourier grid, else fourier grid in same shape
        """
        _, seq_len, hidden_dim = x.shape

        if not self.is_fixed:
            return self.init_fourier_grid(seq_len, hidden_dim)

        if self.encoding.size(-1) == seq_len:
            # 1, s, h
            return self.encoding
        else:
            # 1, ?, h
            return F.interpolate(
                self.encoding.transpose(-1, -2),
                size=seq_len,
            ).transpose(-1, -2)
