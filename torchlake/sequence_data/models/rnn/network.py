import torch
from torch import nn


class RnnCell(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(RnnCell, self).__init__()
        self.input_gate = nn.Linear(input_dim, latent_dim)
        self.memory_gate = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.input_gate(x) + self.memory_gate(h))


class RnnLayer(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, batch_first: bool = True):
        super(RnnLayer, self).__init__()
        self.latent_dim = latent_dim
        self.batch_first = batch_first
        self.cell = RnnCell(input_dim, latent_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        if h is None:
            h = torch.zeros((1, 1, self.latent_dim))

        # recurrent network is suitable on cpu not gpu for sequential operation
        # but for practical, implement in batch, max_seq_len, latent_dim
        if self.batch_first:
            x = x.transpose(0, 1)

        hidden_states = []
        for x_t in x:
            h = self.cell(x_t, h)
            hidden_states.append(h)

        hidden_states = torch.stack(hidden_states, 0)
        if self.batch_first:
            hidden_states = hidden_states.transpose(0, 1)

        return hidden_states
