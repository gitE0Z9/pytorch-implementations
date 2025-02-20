import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        concat_dim = input_dim + latent_dim

        # fused input_gate, forgot_gate, output_gate
        self.w = nn.Linear(concat_dim, 3 * latent_dim)

        # fused memory_gate
        self.memory_gate = nn.Linear(concat_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_tilde = torch.cat([x, h], dim=-1)

        fused_state = self.w(h_tilde).sigmoid()
        memory_state = self.memory_gate(h_tilde).tanh()
        hidden_state, forgot_state, output_state = fused_state.chunk(3, -1)

        c = forgot_state * c + hidden_state * memory_state
        h = output_state * torch.tanh(c)
        return h, c


class LSTMLayer(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, batch_first: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.batch_first = batch_first
        self.cell = LSTMCell(input_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            h = torch.zeros((1, 1, self.latent_dim))
        if c is None:
            c = torch.zeros((1, 1, self.latent_dim))

        # recurrent network is suitable on cpu not gpu for sequential operation
        # loop over in the shape of max_seq_len, batch, latent_dim
        if self.batch_first:
            x = x.transpose(0, 1)

        hidden_states = []
        cell_states = []
        for x_t in x:
            h, c = self.cell(x_t, h, c)
            hidden_states.append(h)
            cell_states.append(c)

        hidden_states = torch.stack(hidden_states, 0)
        cell_states = torch.stack(cell_states, 0)
        if self.batch_first:
            hidden_states = hidden_states.transpose(0, 1)
            cell_states = cell_states.transpose(0, 1)

        return hidden_states, cell_states
