import torch
from torch import nn
from torchlake.common.models import ChannelVector, PositionEncoding1d

from ..vit.model import ViT


class DeiT(ViT):

    def build_foot(self, input_channel: int):
        n = self.image_size // self.patch_size

        self.foot = nn.ModuleDict(
            {
                "pos_embed": PositionEncoding1d(
                    n * n + 2,
                    self.embed_dim,
                    trainable=True,
                ),
                "cls_embed": ChannelVector(self.embed_dim),
                "dist_embed": ChannelVector(self.embed_dim),
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

    def build_head(self, output_size: int):
        self.head = nn.ModuleDict(
            {
                "cls": nn.Linear(self.feature_dim, output_size),
                "dist": nn.Linear(self.feature_dim, output_size),
            }
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # patch embedding
        # b, s, h
        y = self.foot["projection"](x).transpose(-1, -2)

        # cls embedding
        # b, 2+s, h
        cls_embed: torch.Tensor = self.foot["cls_embed"]()
        cls_embed = cls_embed.expand(x.size(0), *cls_embed.shape[1:])

        # dist embedding
        dist_embed: torch.Tensor = self.foot["dist_embed"]()
        dist_embed = dist_embed.expand(x.size(0), *dist_embed.shape[1:])
        y = torch.cat([cls_embed, dist_embed, y], 1)

        # position embedding
        # b, 2+s, h
        y = y + self.foot["pos_embed"](y)

        # b, 2, h
        y = self.neck(y)[:, :2, :]

        # b, o # b, o
        cls_y, dist_y = self.head["cls"](y[:, 0, :]), self.head["dist"](y[:, 1, :])

        if self.training:
            return cls_y, dist_y
        else:
            return (cls_y + dist_y) / 2
