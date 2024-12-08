import torch
import torch.nn.functional as F
from torch import nn


class ClassEmbedding(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.rand(1, 1, embed_dim))

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        return self.embedding


class ViT22BLayer(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm = nn.LayerNorm()
        self.norm_q = nn.LayerNorm()
        self.norm_k = nn.LayerNorm()
        self.attention = nn.MultiheadAttention()
        self.linear = nn.Linear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        q, k, v = self.norm_q(y), self.norm_k(y), y
        att = self.attention(q, k, v)
        y = F.gelu(self.linear(y))

        y = att + y

        return x + y
