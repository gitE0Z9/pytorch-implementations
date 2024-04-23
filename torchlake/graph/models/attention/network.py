import torch
from torch import nn
import torch.nn.functional as F


class GatLayer(nn.Module):
    """ICLR 2018, GRAPH ATTENTION NETWORKS, Cambridge"""

    def __init__(self, in_dim: int, latent_dim: int, num_heads: int = 3):
        super(GatLayer, self).__init__()
        self.input_layer = nn.Linear(in_dim, latent_dim, bias=False)
        self.attention_vectors = nn.Parameter(torch.rand((num_heads, 2 * latent_dim)))

    def get_attention_weight(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """attention weight to neighbors

        Args:
            h (torch.Tensor): node latent tensor, shape is (#edge, 2 * latent_dim)
            mask (torch.Tensor): size of each node's neighborhood(include self), shape is (#node,)

        Returns:
            torch.Tensor: attention weight, shape is (#head, #edge)
        """
        # #head, #edge
        attention_weight = torch.einsum("hf, ef -> he", self.attention_vectors, h)
        attention_weight = F.leaky_relu(attention_weight, 0.2)

        # TODO: masked
        # #head, #edge
        return attention_weight.softmax(dim=-1)

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
            y = h.mean(dim=0).sigmoid()
        else:
            # #head, #node, #latent_dim
            y = h.sigmoid()

        return y

    def forward(
        self,
        x: torch.Tensor,
        edges: torch.Tensor,
        a: torch.Tensor,
        predict: bool = False,
    ) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): node features, shape is (#node, #feature)
            edges (torch.Tensor): edges, shape is (#edge, 2)
            a (torch.Tensor): adjacency matrix, shape is (#node, #node)
            predict (bool, optional): predict layer or not. Defaults to False.

        Returns:
            torch.Tensor: output tensor, shape is (#node, #feature)
        """
        # #node, latent_dim
        h = self.input_layer(x)
        # #edge, 2 * latent_dim
        h = h[edges, :].view(-1, 2 * h.size(-1))

        # degree
        degree = a.sum(0)
        mask = (
            torch.Tensor([d * [i] for i, d in enumerate(degree)]).view(-1).to(a.device)
        )

        # #head, #edge
        attention_weight = self.get_attention_weight(h, mask)
        # #edge, latent_dim
        _, h_j = torch.split(h, 2, -1)

        # TODO: masked
        # #head, latent_dim
        h = torch.einsum("he, ef -> hf", attention_weight, h_j)

        return self.get_output(h, predict)


class GatLayerV2(GatLayer):
    """ICLR 2022, HOW ATTENTIVE ARE GRAPH ATTENTION NETWORKS, Technion, Cambridge"""

    def __init__(self, in_dim: int, latent_dim: int, num_heads: int = 3):
        super(GatLayerV2, self).__init__(in_dim, latent_dim, num_heads)

    def get_attention_weight(self, h: torch.Tensor) -> torch.Tensor:
        """attention weight to neighbors

        Args:
            h (torch.Tensor): node latent tensor, shape is (#edge, 2 * latent_dim)

        Returns:
            torch.Tensor: attention weight, shape is (#head, #edge)
        """
        # #head, #edge
        attention_weight = F.leaky_relu(h, 0.2)
        attention_weight = torch.einsum("hf, ef -> he", self.attention_vectors, h)

        # TODO: masked
        # #head, #edge
        return attention_weight.softmax(dim=-1)
