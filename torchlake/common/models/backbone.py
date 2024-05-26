from typing import Any

import torch
import torch.nn as nn
import torchvision
from ..exceptions import BackboneBuildFailure


class BackboneBase(nn.Module):
    def __init__(
        self,
        finetune_weight: str = "",
        backbone_options: dict[str, Any] = {},
    ):
        super(BackboneBase, self).__init__()
        self.load_backbone(finetune_weight, backbone_options)

    def build_backbone(self, name: str = "") -> nn.Module | None:
        if name:
            backbone_class = getattr(torchvision.models, name)
            backbone = backbone_class(weights="DEFAULT")
            return backbone

        else:
            raise BackboneBuildFailure(name)

    def load_backbone(
        self,
        finetune_weight: str = "",
        backbone_options: dict[str, Any] = {},
    ):
        backbone = self.build_backbone(**backbone_options)

        if finetune_weight:
            backbone.load_state_dict(torch.load(finetune_weight))

        # Remove final classficiation head for
        # 1. reduce parameters
        # 2. get feature from the last layer
        for attr in ["fc", "head"]:
            if getattr(backbone, attr, None):
                setattr(backbone, attr, nn.Identity())

        self.backbone = backbone
