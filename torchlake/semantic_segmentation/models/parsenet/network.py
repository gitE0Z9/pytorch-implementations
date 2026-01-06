from typing import Literal
import torch
from torch import nn

from torchlake.common.models import L2Norm
from torchlake.common.models.vgg_feature_extractor import VGGFeatureExtractor
from ..deeplab.network import deeplab_style_vgg


def parsenet_style_vgg(
    network_name: Literal["vgg11", "vgg13", "vgg16", "vgg19"],
    trainable: bool = True,
) -> VGGFeatureExtractor:
    fe = deeplab_style_vgg(
        network_name,
        trainable,
        large_fov=True,
        pool_5_type="max",
    )

    stage = 0
    for layer in fe.feature_extractor:
        if isinstance(layer, nn.MaxPool2d):
            stage += 1

            # stage 4, 5
            if stage >= 4:
                layer.kernel_size = (3, 3)
                layer.stride = (1, 1)
                layer.padding = (1, 1)

            if stage == 5:
                break

    return fe


class GlobalContextModule(nn.Module):

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        scale: float = 10.0,
    ):
        """Global context module in ParseNet [1506.04579v2]

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            scale (float, optional): init value of output scale. Defaults to 10.0.
        """
        super().__init__()
        self.norm = nn.Sequential(
            L2Norm(input_channel, scale=scale),
            nn.Conv2d(input_channel, output_channel, 1),
        )
        nn.init.xavier_uniform_(self.norm[-1].weight.data)
        nn.init.zeros_(self.norm[-1].bias.data)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            L2Norm(input_channel, scale=scale),
            nn.Dropout(p=0.3),
            nn.Conv2d(input_channel, output_channel, 1),
            # unpool
        )
        nn.init.xavier_uniform_(self.pool[-1].weight.data)
        nn.init.zeros_(self.pool[-1].bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + self.pool(x).repeat(1, 1, *x.shape[2:])
