import torch
from torch import nn


class GruCell(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(GruCell, self).__init__()
        concat_dim = input_dim + latent_dim

        # fused input_gate, output_gate
        self.w = nn.Linear(concat_dim, 2 * latent_dim)

        self.memory_gate_x = nn.Linear(input_dim, latent_dim)
        self.memory_gate_h = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h_tilde = torch.cat([x, h], dim=-1)

        fused_state = self.w(h_tilde).sigmoid()
        hidden_state, output_state = fused_state.chunk(2, -1)
        memory_state = self.memory_gate_x(x) + hidden_state * self.memory_gate_h(h)

        h = output_state * h + (1 - output_state) * memory_state.tanh()
        return h


class GruLayer(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, batch_first: bool = True):
        super(GruLayer, self).__init__()
        self.latent_dim = latent_dim
        self.batch_first = batch_first
        self.cell = GruCell(input_dim, latent_dim)

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
