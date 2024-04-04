import torch
from torch import nn


class RnnCell(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(RnnCell, self).__init__()
        self.input_gate = nn.Linear(input_dim, latent_dim, bias=False)
        self.memory_gate = nn.Linear(latent_dim, latent_dim)
        self.output_gate = nn.Linear(latent_dim, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_state = torch.sigmoid(self.input_gate(x) + self.memory_gate(h))
        return self.output_gate(hidden_state), hidden_state


class RnnLayer(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, batch_first: bool = True):
        super(RnnLayer, self).__init__()
        self.latent_dim = latent_dim
        self.batch_first = batch_first
        self.cell = RnnCell(input_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        hidden_states = []
        h = torch.zeros((1, 1, self.latent_dim))

        # recurrent network is suitable on cpu not gpu for sequential operation
        # but for practical, implement in batch, max_seq_len, latent_dim
        if self.batch_first:
            x = x.transpose(0, 1)

        for x_t in x:
            o, h = self.cell(x_t, h)
            outputs.append(o)
            hidden_states.append(h)

        outputs = torch.stack(outputs, 0).transpose(0, 1)
        hidden_states = torch.stack(hidden_states, 0).transpose(0, 1)

        return outputs, hidden_states
