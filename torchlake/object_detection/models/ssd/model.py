import torch
from torch import nn
from torchlake.common.models.model_base import ModelBase

from ...constants.schema import DetectorContext
from ..base.network import RegHead
from .network import Backbone


class SSD(ModelBase):
    def __init__(self, context: DetectorContext):
        self.context = context
        # 1 is background
        super().__init__(3, context.num_classes + 1)

    def build_foot(self, _):
        self.foot = Backbone()

    def build_head(self, output_size):
        self.head = nn.ModuleList(
            [
                RegHead(512, output_size, 4),
                RegHead(1024, output_size, 6),
                RegHead(512, output_size, 6),
                RegHead(256, output_size, 6),
                RegHead(256, output_size, 4),
                RegHead(256, output_size, 4),
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        loc_list = []
        conf_list = []

        features: list[torch.Tensor] = self.foot(x)
        for feature, head in zip(features, self.head):
            head: RegHead
            loc, conf = head(feature)

            # batch, num anchor?*grid?*grid?, 4
            loc = (
                loc.view(batch_size, head.num_priors, head.coord_dims, -1)
                .permute(0, 1, 3, 2)
                .reshape(batch_size, -1, head.coord_dims)
            )
            # batch, num anchor?*grid?*grid?, num class
            conf = (
                conf.view(batch_size, head.num_priors, head.num_classes, -1)
                .permute(0, 1, 3, 2)
                .reshape(batch_size, -1, head.num_classes)
            )

            loc_list.append(loc)
            conf_list.append(conf)

        # batch, num anchor*grid*grid, 4 # batch, num anchor*grid*grid, num class
        return torch.cat(loc_list, 1), torch.cat(conf_list, 1)
