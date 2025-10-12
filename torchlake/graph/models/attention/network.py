import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax


class GATLayer(nn.Module):
    """ICLR 2018, GRAPH ATTENTION NETWORKS, Cambridge"""

    def __init__(self, in_dim: int, latent_dim: int, num_heads: int = 3):
        super().__init__()
        self.input_layer = nn.Parameter(torch.rand((num_heads, in_dim, latent_dim)))
        self.attention_vector = nn.Parameter(torch.rand((num_heads, 2 * latent_dim)))

    def get_attention_weight(
        self,
        h: torch.Tensor,
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """attention weight to neighbors

        Args:
            h (torch.Tensor): node latent tensor, shape is (#head, #edge, 2 * latent_dim)
            node_indices (torch.Tensor): node indices for each edge, shape is (#edge,)

        Returns:
            torch.Tensor: attention weight, shape is (#head, #edge)
        """
        # #head, #edge
        attention_weight = torch.einsum("hf, hef -> he", self.attention_vector, h)
        attention_weight = F.leaky_relu(attention_weight, 0.2)

        num_heads = self.attention_vector.size(0)

        # #head, #edge
        return scatter_softmax(
            attention_weight,
            node_indices.expand(num_heads, -1),
            dim=-1,
        )

    def get_output(
        self,
        h: torch.Tensor,
        predict: bool = False,
    ) -> torch.Tensor:
        """output as a prediction layer or a hidden layer

        Args:
            h (torch.Tensor): node latent tensor, shape is (#head, #node, #laten_dim)
            predict (bool, optional): predict layer or not. Defaults to False.

        Returns:
            torch.Tensor: _description_
        """
        if predict:
            # #node, #latent_dim
            y = h.mean(dim=0)
        else:
            # #head, #node, #latent_dim
            y = h.permute(1, 0, 2).reshape(h.size(1), -1)

        return y

    def forward(
        self,
        x: torch.Tensor,
        edges: torch.Tensor,
        predict: bool = False,
    ) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): node features, shape is (#node, #feature)
            edges (torch.Tensor): edges, shape is (#edge, 2)
            predict (bool, optional): predict layer or not. Defaults to False.

        Returns:
            torch.Tensor: output tensor, shape is (#node, #feature)
        """
        # #head, #node, latent_dim
        h = torch.einsum("hif, ni -> hnf", self.input_layer, x)
        # #head, #edge, 2, latent_dim
        h = h[:, edges]
        # #head, #edge, 2 * latent_dim
        num_head, num_edge, _, latent_dim = h.shape
        h_concat = h.reshape(num_head, num_edge, 2 * latent_dim)

        node_indices = edges[:, 0]

        # #head, #edge
        attention_weight = self.get_attention_weight(h_concat, node_indices)

        # #head, #edge, 1 x  #head, #edge, latent_dim => #head, #edge, latent_dim
        # #head, #edge, latent_dim => #head, #node, latent_dim
        h = scatter_add(
            attention_weight.unsqueeze(-1) * h[:, :, 1],
            node_indices,
            dim=1,
        )

        # #node, latent_dim
        return self.get_output(h, predict)


class GATLayerV2(GATLayer):
    """ICLR 2022, HOW ATTENTIVE ARE GRAPH ATTENTION NETWORKS, Technion, Cambridge"""

    def __init__(self, in_dim: int, latent_dim: int, num_heads: int = 3):
        super().__init__(in_dim, latent_dim, num_heads)

    def get_attention_weight(
        self,
        h: torch.Tensor,
        node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """attention weight to neighbors

        Args:
            h (torch.Tensor): node latent tensor, shape is (#head, #edge, 2 * latent_dim)
            node_indices (torch.Tensor): node indices for each edge, shape is (#edge,)

        Returns:
            torch.Tensor: attention weight, shape is (#head, #edge)
        """
        # #head, #edge
        attention_weight = F.leaky_relu(h, 0.2)
        attention_weight = torch.einsum(
            "hf, hef -> he",
            self.attention_vector,
            attention_weight,
        )

        num_heads = self.attention_vector.size(0)

        # #head, #edge
        return scatter_softmax(
            attention_weight,
            node_indices.expand(num_heads, -1),
            dim=-1,
        )
