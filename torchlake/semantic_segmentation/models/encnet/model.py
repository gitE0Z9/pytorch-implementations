from __future__ import annotations

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.transforms import CenterCrop

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase
from torchlake.semantic_segmentation.models.encnet.network import EncodingModule2d


class EncNet(ModelBase):
    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        hidden_dim: int = 512,
        output_stride: int = 8,
        k: int = 32,
    ):
        self.hidden_dim = hidden_dim
        self.output_stride = output_stride
        self.k = k
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _: int, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList(
            [
                Conv2dNormActivation(self.foot.hidden_dim_32x, self.hidden_dim, 3),
                EncodingModule2d(self.hidden_dim, self.k),
            ]
        )

    def build_head(self, output_size, **kwargs):
        self.head = nn.ModuleDict(
            {
                "clf": nn.Sequential(
                    nn.Dropout2d(0.1),
                    nn.Conv2d(self.hidden_dim, output_size, 1),
                    nn.Upsample(
                        scale_factor=self.output_stride,
                        mode="bilinear",
                        align_corners=True,
                    ),
                ),
            }
        )

    def build_aux(self):
        h = self.foot.hidden_dim_16x
        self.aux = nn.Sequential(
            Conv2dNormActivation(h, h // 4, 3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(h // 4, self.output_size, 1),
            nn.Upsample(
                scale_factor=self.output_stride,
                mode="bilinear",
                align_corners=True,
            ),
        )

    def build_se_head(self):
        self.head["se"] = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_size),
        )

    def drop_aux(self):
        if hasattr(self, "aux"):
            delattr(self, "aux")

    def drop_se_head(self):
        if "se" in self.head:
            self.head.pop("se")

    def train(self: EncNet, mode: bool = True) -> EncNet:
        result = super().train(mode)

        if not hasattr(self, "aux"):
            self.build_aux()

        if "se" not in self.head:
            self.build_se_head()

        return result

    def forward(
        self,
        x: torch.Tensor,
        output_latent: bool | None = None,
    ) -> torch.Tensor:
        if output_latent is None:
            output_latent = self.training

        features: list[torch.Tensor] = self.foot(x)

        y = self.neck[0](features.pop())
        y = self.neck[1](y, output_latent=output_latent)
        if output_latent:
            y, z = y

        cropper = CenterCrop(x.shape[2:])
        y = self.head["clf"](y)
        y = cropper(y)

        if self.training:
            z = self.head["se"](z)
            aux = self.aux(features.pop())
            aux = cropper(aux)

            if output_latent:
                return y, z, aux
            else:
                return y, aux

        if output_latent:
            return y, z
        else:
            return y
