import torch
from torch import nn
from .network import GatLayer


class Gat(nn.Module):
    """GRAPH ATTENTION NETWORKS, Cambridge"""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        out_dim: int,
        num_heads: int = 3,
        num_layer: int = 4,
    ):
        super(Gat, self).__init__()
        self.layers = nn.Sequential(
            GatLayer(in_dim if index == 0 else latent_dim, latent_dim, num_heads)
            for index in range(num_layer - 1)
        )
        self.fc = GatLayer(latent_dim, out_dim, num_heads)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(x, adj, False)

        return self.fc(y, adj, True)
