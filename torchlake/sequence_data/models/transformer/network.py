import math

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # B, S, S => O(N2)
        scaled_product = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(k.size(-1))

        # masked
        if mask is not None:
            scaled_product *= mask

        # B, S, S
        weights = scaled_product.softmax(-1)

        # B, S, h
        return torch.bmm(weights, v)


class SingleHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.wq = nn.Linear(hidden_dim, output_dim)
        self.wk = nn.Linear(hidden_dim, output_dim)
        self.wv = nn.Linear(hidden_dim, output_dim)
        self.attention = ScaledDotProductAttention()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.attention(self.wq(q), self.wq(k), self.wq(v), mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 1):
        super().__init__()

        # TODO: flexibility (any block) v.s. performance (fusion)
        self.heads = nn.ModuleList(
            SingleHeadAttention(hidden_dim, hidden_dim // num_heads)
            for _ in range(num_heads)
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = torch.concat([head(q, k, v, mask) for head in self.heads], -1)
        return self.fc(y)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attention(x, x, x)
        y = self.dropout(y)
        y = self.norm(y + x)

        y1 = self.fc(y)
        y1 = self.dropout(y1)
        y = self.norm2(y + y1)

        return y


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention2 = MultiHeadAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout1d(p=dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        encoded: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self.attention(x, x, x, mask)
        y = self.dropout(y)
        y = self.norm(y + x)

        # cross attention
        y1 = self.attention2(y, encoded, encoded)
        y1 = self.dropout(y1)
        y = self.norm2(y + y1)

        y2 = self.fc(y)
        y2 = self.dropout(y2)
        return self.norm3(y + y2)
