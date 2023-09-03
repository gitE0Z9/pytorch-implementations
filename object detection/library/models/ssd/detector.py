import torch
from torch import nn

from .network import Backbone, RegHead


class SSD(nn.Module):
    def __init__(self, num_classes: int):
        super(SSD, self).__init__()
        self.backbone = Backbone()
        self.head1 = RegHead(512, 4)
        self.head2 = RegHead(1024, 6)
        self.head3 = RegHead(512, 6)
        self.head4 = RegHead(256, 6)
        self.head5 = RegHead(256, 4)
        self.head6 = RegHead(256, 4)

        self.num_classes = num_classes

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        loc_list = []
        conf_list = []

        f = self.backbone(x)
        for i, k in enumerate(f):
            loc, conf = getattr(self, f"head{i+1}")(k)
            loc_list.append(loc.view(x.size(0), -1))
            conf_list.append(conf.view(x.size(0), -1))

        loc_list = torch.cat(loc_list, 1).view(x.size(0), -1, 4)
        conf_list = torch.cat(conf_list, 1).view(x.size(0), -1, self.num_classes)

        return loc_list, conf_list
