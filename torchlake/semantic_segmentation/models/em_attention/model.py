import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.transforms import CenterCrop

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase

from .network import EMAttention2d


class EMANet(ModelBase):
    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        k: int = 64,
        num_iter: int = 3,
        output_stride: int = 8,
        lambda_a: float = 1,
        momentum: float = 0.9,
    ):
        self.output_stride = output_stride
        self.k = k
        self.num_iter = num_iter
        self.lambda_a = lambda_a
        self.momentum = momentum
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, input_channel: int, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                Conv2dNormActivation(self.foot.hidden_dim_32x, 512, 3),
                EMAttention2d(
                    512,
                    k=self.k,
                    num_iter=self.num_iter,
                    lambda_a=self.lambda_a,
                    momentum=self.momentum,
                ),
            ]
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.Sequential(
            Conv2dNormActivation(512, 256, 3),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, output_size, 1),
            nn.Upsample(
                scale_factor=self.output_stride,
                mode="bilinear",
                align_corners=True,
            ),
        )

    def forward(self, x: torch.Tensor, output_attention: bool = False) -> torch.Tensor:
        y: torch.Tensor = self.foot(x).pop()
        y = self.neck[0](y)
        y = self.neck[1](y, output_attention=output_attention)

        if output_attention:
            y, a = y

        y = self.head(y)
        cropper = CenterCrop(size=x.shape[2:])
        y = cropper(y)

        if output_attention:
            return y, a

        return y
