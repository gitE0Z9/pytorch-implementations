import torch
from torch import nn
import torch.nn.functional as F

from .network import GATLayer, GATLayerV2


class GAT(nn.Module):
    """GRAPH ATTENTION NETWORKS, Cambridge"""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        out_dim: int,
        num_heads: int = 4,
        num_layer: int = 3,
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
        super().__init__()
        layer_class = {
            1: GATLayer,
            2: GATLayerV2,
        }.get(version)
        assert layer_class, "Layer version not supported."

        self.dropout = nn.Dropout(p=0.6)
        self.layers = nn.ModuleList(
            [
                layer_class(
                    in_dim if index == 0 else num_heads * latent_dim,
                    out_dim if index == num_layer - 1 else latent_dim,
                    num_heads,
                )
                for index in range(num_layer)
            ]
        )

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): node features, shape is (#node, latent_dim)
            edges (torch.Tensor): edges, shape is (#edge, 2)

        Returns:
            torch.Tensor: output tensor, shape is (#node, out_dim)
        """
        y = x
        for layer in self.layers[:-1]:
            y = self.dropout(y)
            y = layer(y, edges, predict=False)
            y = F.elu(y, 1)

        return self.layers[-1](y, edges, predict=True)
