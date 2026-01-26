import torch
from torch import nn

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import BoundaryRefinement, GlobalConvolutionBlock
from ..fcn.network import init_deconv_with_bilinear_kernel


class GlobalConvolutionNetwork(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int,
        kernel: int,
    ):
        self.kernel = kernel
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")
        self.foot.fix_target_layers(("1_1", "2_1", "3_1", "4_1"))

    def build_blocks(self, **kwargs):
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    GlobalConvolutionBlock(
                        self.foot.hidden_dim_32x,
                        self.output_size,
                        self.kernel,
                    ),
                    BoundaryRefinement(self.output_size),
                ),
                nn.Sequential(
                    GlobalConvolutionBlock(
                        self.foot.hidden_dim_16x,
                        self.output_size,
                        self.kernel,
                    ),
                    BoundaryRefinement(self.output_size),
                ),
                nn.Sequential(
                    GlobalConvolutionBlock(
                        self.foot.hidden_dim_8x,
                        self.output_size,
                        self.kernel,
                    ),
                    BoundaryRefinement(self.output_size),
                ),
                nn.Sequential(
                    GlobalConvolutionBlock(
                        self.foot.hidden_dim_4x,
                        self.output_size,
                        self.kernel,
                    ),
                    BoundaryRefinement(self.output_size),
                ),
            ]
        )

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.output_size,
                        self.output_size,
                        4,
                        stride=2,
                        padding=1,
                    ),
                ),
                nn.Sequential(
                    BoundaryRefinement(self.output_size),
                    nn.ConvTranspose2d(
                        self.output_size,
                        self.output_size,
                        4,
                        stride=2,
                        padding=1,
                    ),
                ),
                nn.Sequential(
                    BoundaryRefinement(self.output_size),
                    nn.ConvTranspose2d(
                        self.output_size,
                        self.output_size,
                        4,
                        stride=2,
                        padding=1,
                    ),
                ),
                nn.Sequential(
                    BoundaryRefinement(self.output_size),
                    nn.ConvTranspose2d(
                        self.output_size,
                        self.output_size,
                        4,
                        stride=2,
                        padding=1,
                    ),
                ),
            ]
        )

        init_deconv_with_bilinear_kernel(self.neck[0][0])
        init_deconv_with_bilinear_kernel(self.neck[1][1])
        init_deconv_with_bilinear_kernel(self.neck[2][1])
        init_deconv_with_bilinear_kernel(self.neck[3][1])

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            BoundaryRefinement(self.output_size),
            nn.ConvTranspose2d(
                self.output_size,
                self.output_size,
                4,
                stride=2,
                padding=1,
            ),
            BoundaryRefinement(self.output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.foot(x)

        y = self.blocks[0](features.pop())
        for block, neck in zip(self.blocks[1:], self.neck):
            y = block(features.pop()) + neck(y)
        y = self.neck[-1](y)

        return self.head(y)
