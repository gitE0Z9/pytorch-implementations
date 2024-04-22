import torch
from torch import nn
from .network import GatLayer, GatLayerV2


class Gat(nn.Module):
    """GRAPH ATTENTION NETWORKS, Cambridge"""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        out_dim: int,
        num_heads: int = 3,
        num_layer: int = 4,
        version: 1 | 2 = 1,
    ):
        """Graph attention network

        Args:
            in_dim (int): input dimension
            latent_dim (int): latent dimension
            out_dim (int): output dimension
            num_heads (int, optional): number of heads of multi-head-attention. Defaults to 3.
            num_layer (int, optional): number of layers. Defaults to 4.
            version (int, optional): use v1 or v2. Defaults to 1.
        """
        super(Gat, self).__init__()
        mapping = {
            1: GatLayer,
            2: GatLayerV2,
        }
        layer_class = mapping.get(version)
        assert layer_class, "Layer version not supported."

        self.layers = nn.Sequential(
            *[
                layer_class(
                    in_dim if index == 0 else num_heads * latent_dim,
                    latent_dim,
                    num_heads,
                )
                for index in range(num_layer - 1)
            ]
        )
        self.fc = layer_class(num_heads * latent_dim, out_dim, num_heads)

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): node features, shape is (#node, latent_dim)
            edges (torch.Tensor): edges, shape is (#edge, 2)

        Returns:
            torch.Tensor: output tensor, shape is (#node, out_dim)
        """
        for layer in self.layers:
            x = layer(x, edges, False)

        return self.fc(x, edges, True)
