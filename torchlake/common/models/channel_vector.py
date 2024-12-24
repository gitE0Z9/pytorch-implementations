import torch
from torch import nn


class ChannelVector(nn.Module):
    """Class embedding in ViT or BERT, worked as a vector over channels"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.rand(1, 1, embed_dim))

    def forward(self) -> torch.Tensor:
        return self.embedding
