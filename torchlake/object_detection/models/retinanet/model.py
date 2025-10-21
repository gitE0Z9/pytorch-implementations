import torch
from torch import nn
import torch.nn.functional as F
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from ...constants.schema import DetectorContext
from .network import RegHead


class RetinaNet(ModelBase):
    def __init__(self, backbone: ExtractorBase, context: DetectorContext):
        self.context = context
        # 1 is background
        super().__init__(
            3,
            context.num_anchors * (context.num_classes + 1 + 4),
            foot_kwargs={
                "backbone": backbone,
            },
        )

    def build_foot(self, _, **kwargs):
        backbone: ExtractorBase = kwargs.pop("backbone")
        last_dim = backbone.feature_dims[-1]
        self.foot = nn.ModuleDict(
            {
                "backbone": backbone,
                "P6": nn.Conv2d(last_dim, last_dim, 3, stride=2, padding=1),
                "P7": nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(last_dim, last_dim, 3, stride=2, padding=1),
                ),
            }
        )

    def build_blocks(self):
        # P3 ~ P7
        self.blocks = nn.ModuleList(
            (
                nn.Conv2d(d, 256, 1)
                for d in (
                    *self.foot["backbone"].feature_dims[-3:],
                    self.foot["backbone"].feature_dims[-1],
                    self.foot["backbone"].feature_dims[-1],
                )
            )
        )

    def build_neck(self):
        self.neck = lambda x, shape: F.interpolate(x, shape)

    def build_head(self, _):
        # share head
        self.head = RegHead(
            256,
            num_anchors=self.context.num_anchors,
            num_classes=self.context.num_classes,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features: list[torch.Tensor] = self.foot["backbone"](x)
        features.append(self.foot["P6"](features[-1]))
        features.append(self.foot["P7"](features[-1]))

        preds = []
        # P7
        y: torch.Tensor = self.blocks[-1](features.pop())
        preds.append(self.head(y))

        # P6~P3
        shapes = [feature.shape[2:] for feature in features[::-1]]
        for shape, block in zip(shapes, self.blocks[:-1][::-1]):
            y = block(features.pop()) + self.neck(y, shape)
            preds.append(self.head(y))

        # batch size, num anchor * num cell, 5+class across scales
        batch_size = x.size(0)
        preds = tuple(
            pred.view(
                batch_size,
                self.context.num_anchors,
                4 + self.context.num_classes,
                -1,
            )
            .permute(0, 1, 3, 2)
            .flatten(1, 2)
            for pred in preds
        )

        # B, A*H*W, 5+C
        return torch.cat(preds, 1)
