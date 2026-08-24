import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add, scatter_softmax


class GATLayer(nn.Module):

    def __init__(self, input_channel: int, hidden_dim: int, num_heads: int):
        """The layer of Graph attention network v1

        Args:
            input_channel (int): input dimension
            hidden_dim (int): hidden dimension
            num_heads (int, optional): number of heads of multi-head-attention.
        """
        super().__init__()
        self.multi_head_query = nn.Parameter(
            torch.rand((num_heads, input_channel, hidden_dim))
        )
        self.attention_vector = nn.Parameter(torch.rand((num_heads, 2 * hidden_dim)))

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
        predict: bool,
    ) -> torch.Tensor:
        """output as a prediction layer or a hidden layer

        Args:
            h (torch.Tensor): node latent tensor, shape is (#head, #node, hidden_dim)
            predict (bool, optional): predict layer or not.

        Returns:
            torch.Tensor: output, shape is (#node, hidden_dim) or (#node, num_head * hidden_dim)
        """
        if predict:
            # #node, hidden_dim
            y = h.mean(dim=0)
        else:
            _, node_size, _ = h.shape
            # #node, num_head * hidden_dim
            y = h.transpose(0, 1).reshape(node_size, -1)

        return y

    def forward(
        self,
        x: torch.Tensor,
        edges: torch.Tensor,
        predict: bool,
    ) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): node features, shape is (#node, #feature)
            edges (torch.Tensor): edges, shape is (#edge, 2)
            predict (bool, optional): predict layer or not.

        Returns:
            torch.Tensor: output tensor, shape is (#node, hidden_dim) or (#node, num_head * hidden_dim)
        """
        # (#head, input_channel, hidden_dim) x (#node, input_channel) => (#head, #node, hidden_dim)
        h = torch.einsum("hif, ni -> hnf", self.multi_head_query, x)
        # #head, #edge, 2, hidden_dim
        h = h[:, edges]
        num_head, num_edge, _, hidden_dim = h.shape
        # #head, #edge, 2 * hidden_dim
        h_concat = h.reshape(num_head, num_edge, 2 * hidden_dim)

        node_indices = edges[:, 0]

        # #head, #edge
        # TODO: scatter dropout
        attention_weight = self.get_attention_weight(h_concat, node_indices)

        # #head, #edge, 1 x  #head, #edge, hidden_dim => #head, #edge, hidden_dim
        # #head, #edge, hidden_dim => #head, #node, hidden_dim
        h = scatter_add(
            attention_weight.unsqueeze(-1) * h[:, :, 1],
            node_indices,
            dim=1,
        )

        # (#node, hidden_dim) or (#node, num_head * hidden_dim)
        return self.get_output(h, predict)


class GATLayerV2(GATLayer):
    def __init__(self, input_channel: int, hidden_dim: int, num_heads: int):
        """The layer of Graph attention network v2

        Args:
            input_channel (int): input dimension
            hidden_dim (int): hidden dimension
            num_heads (int, optional): number of heads of multi-head-attention.
        """
        super().__init__(input_channel, hidden_dim, num_heads)

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


class Block(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        num_heads: int,
        dropout_prob: float,
        version: 1 | 2,
    ):
        """The block of Graph attention network

        Args:
            input_channel (int): input dimension
            hidden_dim (int): hidden dimension
            num_heads (int, optional): number of heads of multi-head-attention.
            dropout_prob (float, optional): dropout probability.
            version (int, optional): use v1 or v2.
        """
        layer_cls = self._get_layer_class(version)

        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.layer = layer_cls(input_channel, hidden_dim, num_heads)
        self.activation = nn.ELU()

    def _get_layer_class(self, version: 1 | 2 = 1) -> GATLayer | GATLayerV2:
        return {
            1: GATLayer,
            2: GATLayerV2,
        }[version]

    def forward(
        self,
        x: torch.Tensor,
        edges: torch.Tensor,
        predict: bool = False,
    ) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): node features, shape is (#node, input_channel)
            edges (torch.Tensor): edges, shape is (#edge, 2)
            predict (bool, optional): predict layer or not. Defaults to False.

        Returns:
            torch.Tensor: output tensor, shape is (#node, hidden_dim) or (#node, num_head * hidden_dim)
        """
        y = self.dropout(x)
        y = self.layer(y, edges, predict=predict)

        if predict:
            return y

        return self.activation(y)
