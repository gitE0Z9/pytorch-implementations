import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from ...constants.schema import DetectorContext
from .network import RegHead, SPP


class YOLOV3(ModelBase):
    def __init__(
        self,
        backbone: nn.Module,
        context: DetectorContext,
        hidden_dim_8x: int,
        hidden_dim_16x: int,
        hidden_dim_32x: int,
        enable_spp: bool = True,
    ):
        self.context = context
        self.hidden_dim_8x = hidden_dim_8x
        self.hidden_dim_16x = hidden_dim_16x
        self.hidden_dim_32x = hidden_dim_32x
        super().__init__(
            3,
            1,
            foot_kwargs={
                "backbone": backbone,
            },
            blocks_kwargs={
                "enable_spp": enable_spp,
            },
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    # from 32x
                    Conv2dNormActivation(
                        self.hidden_dim_32x,
                        self.hidden_dim_32x // 2,
                        1,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim_32x // 2,
                        self.hidden_dim_32x,
                        3,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim_32x,
                        self.hidden_dim_32x // 2,
                        1,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    # spp position
                    Conv2dNormActivation(
                        self.hidden_dim_32x // 2,
                        self.hidden_dim_32x,
                        3,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim_32x,
                        self.hidden_dim_32x // 2,
                        1,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                ),
                nn.Sequential(
                    # from block 0
                    Conv2dNormActivation(
                        self.hidden_dim_32x // 2,
                        self.hidden_dim_16x // 2,
                        1,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    nn.Upsample(scale_factor=2),
                    # concat from 61, from 16x
                ),
                nn.Sequential(
                    Conv2dNormActivation(
                        self.hidden_dim_16x // 2,
                        self.hidden_dim_8x // 2,
                        1,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    nn.Upsample(scale_factor=2),
                    # concat from 36, from 8x
                ),
            ]
        )

        enable_spp: bool = kwargs.pop("enable_spp")
        if enable_spp:
            self.blocks[0].insert(3, SPP())
            self.blocks[0].insert(
                4,
                Conv2dNormActivation(
                    (self.hidden_dim_32x // 2) * 4,
                    self.hidden_dim_32x // 2,
                    1,
                    activation_layer=lambda: nn.LeakyReLU(0.1),
                    inplace=None,
                ),
            )

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                nn.Sequential(
                    # from block 1
                    Conv2dNormActivation(
                        self.hidden_dim_16x // 2 + self.hidden_dim_16x,
                        self.hidden_dim_16x // 2,
                        1,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim_16x // 2,
                        self.hidden_dim_16x,
                        3,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim_16x,
                        self.hidden_dim_16x // 2,
                        1,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim_16x // 2,
                        self.hidden_dim_16x,
                        3,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim_16x,
                        self.hidden_dim_16x // 2,
                        1,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                ),
                nn.Sequential(
                    # from block 2
                    Conv2dNormActivation(
                        self.hidden_dim_8x // 2 + self.hidden_dim_8x,
                        self.hidden_dim_8x // 2,
                        1,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim_8x // 2,
                        self.hidden_dim_8x,
                        3,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim_8x,
                        self.hidden_dim_8x // 2,
                        1,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim_8x // 2,
                        self.hidden_dim_8x,
                        3,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                    Conv2dNormActivation(
                        self.hidden_dim_8x,
                        self.hidden_dim_8x // 2,
                        1,
                        activation_layer=lambda: nn.LeakyReLU(0.1),
                        inplace=None,
                    ),
                ),
            ]
        )

    def build_head(self, _):
        self.head = nn.ModuleList(
            [
                # from block 0
                RegHead(
                    self.hidden_dim_32x // 2,
                    self.context.num_anchors[0],
                    self.context.num_classes,
                ),
                # from neck 0
                RegHead(
                    self.hidden_dim_16x // 2,
                    self.context.num_anchors[1],
                    self.context.num_classes,
                ),
                # from neck 1
                RegHead(
                    self.hidden_dim_8x // 2,
                    self.context.num_anchors[2],
                    self.context.num_classes,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = self.foot(x)

        head = self.head[0]
        y = self.blocks[0](features.pop())
        outputs = [head(y)]
        for block, neck, head in zip(self.blocks[1:], self.neck, self.head[1:]):
            y = block(y)
            y = neck(torch.cat([y, features.pop()], 1))
            # B, A*C, H, W
            outputs.append(head(y))

        # from 8x to 32x
        reversed(outputs)

        # 3 x (B, A*C, H, W)
        return outputs
