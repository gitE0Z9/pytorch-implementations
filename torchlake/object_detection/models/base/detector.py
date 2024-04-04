from typing import Any

import torch
import torch.nn as nn
import torchvision
from .exceptions import BackboneBuildFailure


class DetectorBase(nn.Module):
    def __init__(
        self,
        finetune_weight: str = "",
        backbone_options: dict[str, Any] = {},
    ):
        super(DetectorBase, self).__init__()
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

        if getattr(backbone, "fc", None):
            backbone.fc = nn.Identity()
        elif getattr(backbone, "head", None):
            backbone.head = nn.Identity()

        self.backbone = backbone
