import torch
from torch import nn


class LstmCell(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(LstmCell, self).__init__()
        self.input_gate_x = nn.Linear(input_dim, latent_dim)
        self.memory_gate_x = nn.Linear(input_dim, latent_dim)
        self.forgot_gate_x = nn.Linear(input_dim, latent_dim)
        self.output_gate_x = nn.Linear(input_dim, latent_dim)

        self.input_gate_h = nn.Linear(latent_dim, latent_dim)
        self.memory_gate_h = nn.Linear(latent_dim, latent_dim)
        self.forgot_gate_h = nn.Linear(latent_dim, latent_dim)
        self.output_gate_h = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_state = torch.sigmoid(self.input_gate_x(x) + self.input_gate_h(h))
        forgot_state = torch.sigmoid(self.memory_gate_x(x) + self.memory_gate_h(h))
        memory_state = torch.tanh(self.memory_gate_x(x) + self.memory_gate_h(h))
        output_state = torch.sigmoid(self.output_gate_x(x) + self.output_gate_h(h))

        c = forgot_state * c + hidden_state * memory_state
        h = output_state * torch.tanh(c)
        return h, c


class LstmLayer(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, batch_first: bool = True):
        super(LstmLayer, self).__init__()
        self.latent_dim = latent_dim
        self.batch_first = batch_first
        self.cell = LstmCell(input_dim, latent_dim)

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
        # but for practical, implement in batch, max_seq_len, latent_dim
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
