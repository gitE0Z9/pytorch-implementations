from itertools import pairwise
from typing import Literal

import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.flatten import FlattenFeature
from torchlake.common.models.model_base import ModelBase

from ...constants.schema import DetectorContext
from ..yolov3.network import RegHead


class YOLOV3Tiny(ModelBase):
    def __init__(
        self,
        context: DetectorContext,
        dropout_prob: float = 0.5,
        head_type: Literal["classification", "detection"] = "detection",
    ):
        """tiny YOLOV3

        Args:
            context (DetectorContext): detector context
            dropout_prob (float, optional): dropout prob. Defaults to 0.5.
        """
        self.dropout_prob = dropout_prob
        self.head_type = head_type
        self.context = context
        super().__init__(
            3,
            self.context.num_classes if head_type == "classification" else 1,
        )

    @property
    def feature_dim(self) -> int:
        return 1024

    @property
    def config(self) -> list[int]:
        return [16, 32, 64, 128, 256, 512, 1024]

    def build_foot(self, input_channel):
        self.foot = nn.Sequential(
            Conv2dNormActivation(
                input_channel,
                16,
                3,
                activation_layer=lambda: nn.LeakyReLU(0.1),
                inplace=None,
            ),
            nn.MaxPool2d(2, 2),
        )

    def build_blocks(self):
        blocks = []

        for i, (in_c, out_c) in enumerate(pairwise(self.config)):
            blocks.append(
                Conv2dNormActivation(
                    in_c,
                    out_c,
                    3,
                    activation_layer=lambda: nn.LeakyReLU(0.1),
                    inplace=None,
                )
            )

            if i <= len(self.config) - 3:
                blocks.append(nn.MaxPool2d(2, 2))

        self.blocks = nn.Sequential(*blocks)

    def build_neck(self):
        if self.head_type == "classification":
            self.neck = nn.Dropout(self.dropout_prob)
        else:
            self.neck = nn.ModuleList(
                [
                    nn.Sequential(
                        Conv2dNormActivation(
                            self.config[-1],
                            256,
                            3,
                            activation_layer=lambda: nn.LeakyReLU(0.1),
                            inplace=None,
                        )
                    ),
                    nn.Sequential(
                        Conv2dNormActivation(
                            256,
                            128,
                            3,
                            activation_layer=lambda: nn.LeakyReLU(0.1),
                            inplace=None,
                        ),
                        nn.UpsamplingNearest2d(size=self.context.grid_sizes[0]),
                    ),
                ]
            )

    def build_head(self, output_size: int):
        if self.head_type == "classification":
            self.head = nn.Sequential(
                FlattenFeature(reduction="mean"),
                nn.Linear(self.feature_dim, output_size),
            )
        else:
            self.head = nn.ModuleList(
                [
                    RegHead(
                        256,
                        self.context.num_anchors[0],
                        self.context.num_classes,
                    ),
                    RegHead(
                        256,
                        self.context.num_anchors[1],
                        self.context.num_classes,
                    ),
                ]
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        y = self.foot(x)

        features = []
        pool_count = 0
        for block in self.blocks:
            y = block(y)

            if isinstance(block, nn.MaxPool2d):
                pool_count += 1
                if pool_count == 3:
                    features.append(y)
        features.append(y)

        y = self.neck[0](features.pop())
        outputs = [self.head[0](y)]
        y = self.neck[1](y)
        y = self.head[1](torch.cat([y, features.pop()], 1))
        # B, A*C, H, W
        outputs.append(y)

        # 16x, 64x
        # 2 x (B, A*C, H, W)
        return outputs[::-1]
