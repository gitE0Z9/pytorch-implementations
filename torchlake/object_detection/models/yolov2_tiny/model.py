from itertools import pairwise
from typing import Literal
import torch.nn as nn
from torchlake.common.models.flatten import FlattenFeature
from torchlake.common.models.model_base import ModelBase
from torchvision.ops import Conv2dNormActivation

from ...constants.schema import DetectorContext


class YOLOV2Tiny(ModelBase):
    def __init__(
        self,
        context: DetectorContext,
        dropout_prob: float = 0.5,
        head_type: Literal["classification", "detection"] = "detection",
    ):
        """tiny YOLOV1

        Args:
            context (DetectorContext): detector context
            dropout_prob (float, optional): dropout prob. Defaults to 0.5.
        """
        self.dropout_prob = dropout_prob
        self.head_type = head_type
        self.context = context
        super().__init__(
            3,
            (
                context.num_anchors * 5 + context.num_classes
                if head_type == "detection"
                else context.num_classes
            ),
        )

    @property
    def feature_dim(self) -> int:
        return 512

    @property
    def config(self) -> list[int]:
        return [16, 32, 64, 128, 256, 512, 1024, 512]

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

            if i < len(self.config) - 3:
                blocks.append(nn.MaxPool2d(2, 2))

        self.blocks = nn.Sequential(*blocks)

    def build_neck(self):
        self.neck = nn.Dropout(self.dropout_prob)

    def build_head(self, output_size: int):
        if self.head_type == "classification":
            self.head = nn.Sequential(
                FlattenFeature(reduction=None),
                nn.Linear(self.feature_dim, output_size),
            )

        elif self.head_type == "detection":
            self.head = nn.Sequential(
                nn.Conv2d(self.feature_dim, output_size, 1),
            )
