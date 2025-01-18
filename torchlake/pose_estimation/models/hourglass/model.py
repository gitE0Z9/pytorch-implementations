import torch
from torch import nn
from torchlake.common.models import ResBlock
from torchlake.common.models.model_base import ModelBase
from torchvision.ops import Conv2dNormActivation

from .network import AuxiliaryHead, BottleNeck, Hourglass2d


class StackedHourglass(ModelBase):

    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        hidden_dim: int = 256,
        num_stack: int = 8,
        num_nested: int = 4,
        num_resblock: int = 1,
    ):
        self.hidden_dim = hidden_dim
        self.num_stack = num_stack
        self.num_nested = num_nested
        self.num_resblock = num_resblock
        super().__init__(input_channel, output_size)

    def build_foot(self, input_channel):
        self.foot = nn.Sequential(
            Conv2dNormActivation(input_channel, 64, 7, stride=2),
            ResBlock(64, 128, block=BottleNeck(64, 64)),
            nn.MaxPool2d(2, 2),
            ResBlock(128, 128, block=BottleNeck(128, 64)),
            ResBlock(
                128,
                self.hidden_dim,
                block=BottleNeck(128, self.hidden_dim // 2),
            ),
        )

    def build_blocks(self):
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Hourglass2d(self.hidden_dim, self.num_nested, self.num_resblock),
                    *[
                        ResBlock(
                            self.hidden_dim,
                            self.hidden_dim,
                            block=BottleNeck(self.hidden_dim, self.hidden_dim // 2),
                        )
                        for _ in range(self.num_resblock)
                    ],
                )
                for _ in range(self.num_stack)
            ]
        )

    def build_head(self, output_size):
        self.head = nn.ModuleList(
            [AuxiliaryHead(self.hidden_dim, output_size) for _ in range(self.num_stack)]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        z = self.foot(x)

        outputs = []
        for block, head in zip(self.blocks[:-1], self.head[:-1]):
            w = block(z)
            y, w = head(w, output_neck=True)
            z = z + w
            outputs.append(y)

        w = self.blocks[-1](z)
        y = self.head[-1](w)
        outputs.append(y)

        # B, num_stack, C, H, W
        return torch.stack(outputs, dim=1)
