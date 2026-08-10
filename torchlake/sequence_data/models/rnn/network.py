import torch
from torch import nn


class RNNCell(nn.Module):

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        # fused input data & memory gate
        self.w = nn.Linear(input_dim + latent_dim, latent_dim)
        self.activation = activation or nn.ReLU()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h_tilde = torch.cat([x, h], dim=-1)
        return self.activation(self.w(h_tilde))


class RNNLayer(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, batch_first: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.batch_first = batch_first
        self.cell = RNNCell(input_dim, latent_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        if h is None:
            batch_size = x.size(0) if self.batch_first else x.size(1)
            h = torch.zeros((batch_size, self.latent_dim), device=x.device)

        # recurrent network is suitable on cpu not gpu for sequential operation
        # loop over in the shape of max_seq_len, batch, latent_dim
        if self.batch_first:
            x = x.transpose(0, 1)

        hidden_states = []
        for x_t in x:
            h = self.cell(x_t, h)
            hidden_states.append(h)

        # S x (B, h) => S, B, h
        hidden_states = torch.stack(hidden_states, 0)
        if self.batch_first:
            hidden_states = hidden_states.transpose(0, 1)

        return hidden_states
