import torch
from torch import nn
from torchlake.common.models.model_base import ModelBase

from ...constants.schema import DetectorContext
from .network import Backbone, RegHead


class SSD(ModelBase):
    def __init__(self, context: DetectorContext, trainable: bool = False):
        self.context = context
        self.trainable = trainable
        # 1 is background
        super().__init__(3, context.num_classes + 1)

    def build_foot(self, _):
        self.foot = Backbone(trainable=self.trainable)

    def build_head(self, output_size):
        self.head = nn.ModuleList(
            [
                RegHead(512, 4, output_size),
                RegHead(1024, 6, output_size),
                RegHead(512, 6, output_size),
                RegHead(256, 6, output_size),
                RegHead(256, 4, output_size),
                RegHead(256, 4, output_size),
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        y = []

        features: list[torch.Tensor] = self.foot(x)
        for feature, head in zip(features, self.head):
            head: RegHead
            # batch, num anchor?*grid?*grid?, 4 + 1 + num class
            y_i: torch.Tensor = (
                head(feature)
                .reshape(
                    batch_size,
                    head.num_priors,
                    head.coord_dims + head.num_classes,
                    -1,
                )
                .permute(0, 1, 3, 2)
                .reshape(batch_size, -1, head.coord_dims + head.num_classes)
            )

            y.append(y_i)

        # batch, num anchor*grid*grid, 4 + 1 + num class
        return torch.cat(y, 1)
