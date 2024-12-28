from typing import Literal

import torch
from torch import nn
from torchlake.common.models import ChannelVector, PositionEncoding1d
from torchlake.common.models.model_base import ModelBase


class ViT(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_head: int = 8,
        num_encoder_layers: int = 6,
        size: Literal["tiny", "small", "base", "large", "huge"] | None = None,
    ):
        if size is not None:
            self.size = size
            num_head, num_encoder_layers, embed_dim, hidden_dim = self.config

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.num_encoder_layers = num_encoder_layers
        super().__init__(input_channel, output_size)

    @property
    def feature_dim(self) -> int:
        return self.embed_dim

    @property
    def config(self) -> list[list[int]]:
        # num_head, num_encoder_layers, embed_dim, hidden_dim
        return {
            "tiny": [3, 12, 192, 768],
            "small": [6, 12, 384, 1536],
            "base": [12, 12, 768, 3072],
            "large": [16, 24, 1024, 4096],
            "huge": [16, 32, 1280, 5120],
        }[self.size]

    def build_foot(self, input_channel: int):
        n = self.image_size // self.patch_size

        self.foot = nn.ModuleDict(
            {
                "pos_embed": PositionEncoding1d(
                    n * n + 1,
                    self.embed_dim,
                    trainable=True,
                ),
                "cls_embed": ChannelVector(self.embed_dim),
                "projection": nn.Sequential(
                    nn.Conv2d(
                        input_channel,
                        self.embed_dim,
                        kernel_size=self.patch_size,
                        stride=self.patch_size,
                    ),
                    nn.Flatten(2, -1),
                ),
            }
        )

    def build_neck(self):
        self.neck = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.embed_dim,
                self.num_head,
                dim_feedforward=self.hidden_dim,
                batch_first=True,
                norm_first=True,
            ),
            self.num_encoder_layers,
        )

    def build_head(self, output_size: int):
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # patch embedding
        # b, s, h
        y = self.foot["projection"](x).transpose(-1, -2)

        # cls embedding
        # b, 1+s, h
        cls_embed: torch.Tensor = self.foot["cls_embed"]()
        cls_embed = cls_embed.expand(x.size(0), *cls_embed.shape[1:])
        y = torch.cat([cls_embed, y], 1)

        # position embedding
        # b, 1+s, h
        y = y + self.foot["pos_embed"](y)

        y = self.neck(y)[:, 0, :]

        # paper: a MLP with one hidden layer at pre-training
        # time and by a single linear layer at fine-tuning time
        y = self.head(y)

        return y
