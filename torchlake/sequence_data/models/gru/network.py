import torch
from torch import nn


class GRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # fused input_gate, reset_gate, memory_gate
        self.w_is = nn.Linear(input_dim, 3 * hidden_dim)
        self.w_ss = nn.Linear(hidden_dim, 3 * hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        seq_len, _, _ = x.shape

        # s, b, h
        z_is = self.w_is(x)

        hidden_states = []
        for t in range(seq_len):
            z_ss = self.w_ss(h)
            hidden_state, reset_state = (
                (z_is[t, :, : -self.hidden_dim] + z_ss[:, : -self.hidden_dim])
                .sigmoid()
                .chunk(2, -1)
            )
            memory_state = (
                z_is[t, :, -self.hidden_dim :]
                + hidden_state * z_ss[:, -self.hidden_dim :]
            ).tanh()

            h = reset_state * h + (1 - reset_state) * memory_state
            hidden_states.append(h)

        # S x (B, h) => S, B, h
        return torch.stack(hidden_states, 0)


class GRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        batch_first: bool,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # D
        self.factor = 2 if bidirectional else 1
        self.cells = nn.ModuleList(
            [
                GRUCell(
                    input_dim if l == 0 else self.factor * hidden_dim,
                    hidden_dim,
                )
                for l in range(num_layers)
            ]
        )

        if bidirectional:
            self.reverse_cells = nn.ModuleList(
                [
                    GRUCell(
                        input_dim if l == 0 else self.factor * hidden_dim,
                        hidden_dim,
                    )
                    for l in range(num_layers)
                ]
            )

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        if self.batch_first:
            x = x.transpose(0, 1)

        # s, b, d
        _, batch_size, _ = x.shape
        if h is None:
            # b, L*D*h
            h = torch.zeros(
                (batch_size, self.num_layers * self.factor * self.hidden_dim),
                device=x.device,
            )

        hidden_states = []
        for l in range(self.num_layers):
            # left
            cell = self.cells[l]
            s = l * self.factor * self.hidden_dim
            e = s + self.hidden_dim

            # s, b, h
            hidden_states_l = cell(x, h[:, s:e])
            # L x (s, b, h)
            hidden_states.append(hidden_states_l)

            # right
            if self.bidirectional:
                reverse_cell = self.reverse_cells[l]
                s, e = e, e + self.hidden_dim

                # s, b, h
                hidden_states_r = reverse_cell(x.flip(0), h[:, s:e]).flip(0)
                # L x (s, b, h)
                hidden_states.append(hidden_states_r)

                # s, b, D*h
                x = torch.cat((hidden_states_l, hidden_states_r), -1)
            else:
                # s, b, h
                x = hidden_states_l

        # s, b, L*D*h
        hidden_states = torch.cat(hidden_states, -1)
        if self.batch_first:
            # b, s, L*D*h
            hidden_states = hidden_states.transpose(0, 1)

        return hidden_states
