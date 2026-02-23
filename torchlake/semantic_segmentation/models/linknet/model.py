import torch
from torch import nn

from torchlake.common.models import ConvBNReLU
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase
from torchvision.ops import Conv2dNormActivation
from .network import DecoderBlock, EncoderBlock


class LinkNet(ModelBase):
    def __init__(self, backbone: ExtractorBase, output_size: int):
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")
        self.foot.feature_extractor[4][0].conv1.stride = (2, 2)
        self.foot.feature_extractor[4][0].downsample = nn.Sequential(
            nn.Conv2d(
                self.foot.hidden_dim_4x,
                self.foot.hidden_dim_4x,
                1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(self.foot.hidden_dim_4x),
        )

    # def __init__(self, input_channel: int, output_size: int):
    #     super().__init__(input_channel, output_size)

    # def build_foot(self, input_channel: int, **kwargs):
    #     self.foot = nn.Sequential(
    #         Conv2dNormActivation(input_channel, 64, 7, stride=2),
    #         nn.MaxPool2d(3, 2, padding=1),
    #     )

    # def build_blocks(self, **kwargs):
    #     # from shallow to deep
    #     self.blocks = nn.ModuleList(
    #         [
    #             EncoderBlock(64, 64),
    #             EncoderBlock(64, 128),
    #             EncoderBlock(128, 256),
    #             EncoderBlock(256, 512),
    #         ]
    #     )

    def build_neck(self, **kwargs):
        # from deep to shallow
        self.neck = nn.ModuleList(
            [
                DecoderBlock(self.foot.hidden_dim_32x, self.foot.hidden_dim_16x),
                DecoderBlock(self.foot.hidden_dim_16x, self.foot.hidden_dim_8x),
                DecoderBlock(self.foot.hidden_dim_8x, self.foot.hidden_dim_4x),
                DecoderBlock(self.foot.hidden_dim_4x, self.foot.hidden_dim_4x),
                # DecoderBlock(512, 256),
                # DecoderBlock(256, 128),
                # DecoderBlock(128, 64),
                # DecoderBlock(64, 64),
            ]
        )

    def build_head(self, output_size, **kwargs):
        h = self.foot.hidden_dim_4x // 2

        self.head = nn.Sequential(
            ConvBNReLU(
                self.foot.hidden_dim_4x,
                h,
                4,
                padding=1,
                stride=2,
                deconvolution=True,
            ),
            ConvBNReLU(h, h, 3, padding=1),
            ConvBNReLU(h, output_size, 2, stride=2, deconvolution=True),
            # ConvBNReLU(64, 32, 4, padding=1, stride=2, deconvolution=True),
            # ConvBNReLU(32, 32, 3, padding=1),
            # ConvBNReLU(32, output_size, 2, stride=2, deconvolution=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = self.foot(x)
        # features = []
        # for block in self.blocks:
        #     y = block(y)
        #     features.append(y)
        features: list[torch.Tensor] = self.foot(x)

        y = self.neck[0](features.pop())
        for neck in self.neck[1:]:
            y = y + features.pop()
            y = neck(y)

        return self.head(y)
