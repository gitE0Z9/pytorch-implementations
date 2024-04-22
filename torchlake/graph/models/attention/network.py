import torch
from torch import nn
from torch.nn.functional import leaky_relu_


class GatLayer(nn.Module):
    """ICLR 2018, GRAPH ATTENTION NETWORKS, Cambridge"""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 3):
        super(GatLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attention_layers = [
            nn.Linear(out_dim * 2, out_dim * 2, bias=False) for _ in range(num_heads)
        ]

    def get_attention_weight(
        self,
        attention_layer: nn.Module,
        h: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        attention_weight = attention_layer(h) * adj
        leaky_relu_(attention_weight)

        return attention_weight.softmax(dim=0)

    def get_multihead_output(
        self,
        h_primes: list[torch.Tensor],
        predict: bool = False,
    ) -> torch.Tensor:
        if predict:
            y = torch.stack(h_primes, dim=1)
            y = y.mean(dim=1).sigmoid()
        else:
            y = h_primes.sigmoid()
            y = torch.stack(y, dim=1)

        return y

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        predict: bool = False,
    ) -> torch.Tensor:
        h = self.linear(x)
        concated_h = torch.cat([h, h], -1)

        h_primes = []
        for attention_layer in self.attention_layers:
            attention_weight = self.get_attention_weight(
                attention_layer, concated_h, adj
            )
            h_prime = torch.mm(attention_weight, h)
            h_primes.append(h_prime)

        return self.get_multihead_output(h_primes, predict)
