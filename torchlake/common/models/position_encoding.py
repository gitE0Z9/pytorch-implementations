import torch
from torch import nn
from ..utils.numerical import generate_grid


class PositionEncoding(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, hidden_dim = x.shape

        grid_x, grid_y = generate_grid(hidden_dim, seq_len)
        grid_x = grid_x[:, 0::2].repeat_interleave(2, 1)
        denominator = 10000 ** (2 * grid_x / hidden_dim)
        y = grid_y / denominator

        y[:, 0::2] = y[:, 0::2].sin()
        y[:, 1::2] = y[:, 1::2].cos()

        # 1, S, h
        return y.unsqueeze(0).to(x.device)
