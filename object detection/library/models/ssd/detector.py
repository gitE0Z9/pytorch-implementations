import torch
from torch import nn
import torchvision
from models.ssd.network import Backbone, RegHead


class Ssd(nn.Module):
    def __init__(
        self,
        num_classes: int,
        finetune_weight: str = "",
    ):
        super(Ssd, self).__init__()
        self.load_backbone(finetune_weight)

        self.head1 = RegHead(512, num_classes, 4)
        self.head2 = RegHead(1024, num_classes, 6)
        self.head3 = RegHead(512, num_classes, 6)
        self.head4 = RegHead(256, num_classes, 6)
        self.head5 = RegHead(256, num_classes, 4)
        self.head6 = RegHead(256, num_classes, 4)

        self.num_classes = num_classes

    def load_backbone(self, finetune_weight: str = ""):
        self.backbone = Backbone()
        if finetune_weight:
            self.backbone.load_state_dict(torch.load(finetune_weight))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc_list = []
        conf_list = []

        features = self.backbone(x)
        for i, feature in enumerate(features):
            loc, conf = getattr(self, f"head{i+1}")(feature)
            loc_list.append(loc.view(x.size(0), -1))
            conf_list.append(conf.view(x.size(0), -1))

        loc_list = torch.cat(loc_list, 1).view(x.size(0), -1, 4)
        conf_list = torch.cat(conf_list, 1).view(x.size(0), -1, self.num_classes)

        return loc_list, conf_list
