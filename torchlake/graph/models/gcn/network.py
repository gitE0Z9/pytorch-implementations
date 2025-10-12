import torch
from torch import nn

from torchlake.common.utils.sparse import eye_matrix


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def get_topology_transform(self, a: torch.Tensor) -> torch.Tensor:
        node_size, _ = a.shape

        a_tilde = (eye_matrix(node_size).to(a.device) + a).float()
        d_tilde = a_tilde.sum(dim=-1).float()  # pow need input type float
        D = torch.sparse_coo_tensor(
            torch.arange(node_size).repeat(2, 1).to(a.device),
            d_tilde.pow(-0.5).to_dense(),
        )
        a_hat = D * a_tilde * D

        return a_hat

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Applies a Graph Convolutional Network (GCN) layer.

        Args:
            x (torch.Tensor): Input features with shape (Node, in_dim)
            a (torch.Tensor): Adjacency matrix with shape (Node, Node)

        Returns:
            torch.Tensor: Output features with shape (Node, out_dim)
        """
        H = self.get_topology_transform(a)

        return torch.mm(H, self.linear(x))


class GCNResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.layer = GCNLayer(in_dim, out_dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        y = self.layer(x, a)

        return y + x
