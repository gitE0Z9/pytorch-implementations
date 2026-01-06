# plain: bi_w = 5, bi_xy_std = 50, bi_rgb_std = 10, pos_w = 3, pos_xy_std = 3.
# multiscale: bi_w = 3, bi_xy_std = 95, bi_rgb_std = 3, pos_w = 3, pos_xy_std = 3.
# coco: bi_w = 5, bi_xy_std = 67, bi_rgb_std = 3, pos_w = 3, pos_xy_std = 3.
# FOV: bi_w = 4, bi_xy_std = 121, bi_rgb_std = 5, pos_w = 3, pos_xy_std = 3
# multiscale + FOV: bi_w = 4, bi_xy_std = 83, bi_rgb_std = 5, pos_w = 3, pos_xy_std = 3
# COCO FOV: bi_w = 5, bi_xy_std = 69, bi_rgb_std = 5, pos_w = 3, pos_xy_std = 3
# multiscale + FOV + COCO: bi_w = 4, bi_xy_std = 59, bi_rgb_std = 5, pos_w = 3, pos_xy_std = 3

import torch
from annotated_types import T
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.transforms import CenterCrop

from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase


class DeepLab(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        output_size: int = 1,
        large_fov: bool = True,
    ):
        """DeepLab v1 in paper [1412.7062v4]

        Args:
            backbone (ExtractorBase): feature extractor.
            output_size (int, optional): output size. Defaults to 1.
            large_fov (bool, optional): use larger field of view. Defaults to True.
        """
        self.fov_dim = 1024 if large_fov else 4096
        self.large_fov = large_fov
        super().__init__(
            backbone.input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot: ExtractorBase = kwargs.pop("backbone")
        self.foot.fix_target_layers(("1_1", "2_1", "3_1", "4_1", "6_1"))

    def build_blocks(self, **kwargs):
        layer = Conv2dNormActivation(
            self.fov_dim,
            self.output_size,
            1,
            norm_layer=None,
        )
        nn.init.normal_(layer[0].weight.data, 0, 0.01)
        nn.init.zeros_(layer[0].bias.data)

        self.blocks = nn.Sequential(layer)

    def build_neck(self, **kwargs):
        self.neck = nn.ModuleList([])
        for i, dim in enumerate(
            (
                self.foot.hidden_dim_16x,
                self.foot.hidden_dim_8x,
                self.foot.hidden_dim_4x,
                self.foot.hidden_dim_2x,
                self.input_channel,
            )
        ):
            layer = nn.Sequential(
                Conv2dNormActivation(
                    dim,
                    128,
                    3,
                    norm_layer=None,
                ),
                nn.Dropout(0.5),
                Conv2dNormActivation(
                    128,
                    128,
                    1,
                    norm_layer=None,
                ),
                nn.Dropout(0.5),
                nn.Conv2d(128, self.output_size, 1),
            )
            nn.init.normal_(layer[0][0].weight.data, 0, 0.001)
            nn.init.zeros_(layer[0][0].bias.data)

            nn.init.normal_(layer[2][0].weight.data, 0, 0.005 if i == 0 else 0.001)
            nn.init.zeros_(layer[2][0].bias.data)

            if i != 1:
                nn.init.normal_(layer[4].weight.data, 0, 0.01)
                nn.init.zeros_(layer[4].bias.data)

            self.neck.append(layer)

    def build_head(self, output_size, **kwargs):
        self.head = None

    def train(self: T, mode: bool = True) -> T:
        result = super().train(mode)

        # every feature map downsample to 8x
        for block, s in zip(self.neck, (1, 1, 2, 4, 8)):
            # only modify first encountered conv2dnormactivation
            block[0][0].stride = (s, s)

        # no upsampling
        # result.upsamples = nn.ModuleList([nn.Identity()] * 5)

        return result

    def eval(self: T) -> T:
        result = super().eval()

        # no downsample for multi scale prediction
        for block in result.neck:
            # only modify first encountered conv2dnormactivation
            block[0][0].stride = (1, 1)

        # upsample every prediction
        result.head = nn.ModuleList(
            [
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                for scale_factor in [8, 8, 8, 4, 2]
            ]
        )

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        during training, output is 8x downsampled logit map and label is 8x bilinear downsampled
        during inference, output is 8x downsampled logit map and 8x bilinear upsampled

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: logit map
        """
        is_eval = not self.training
        if is_eval:
            cropper = CenterCrop(x.shape[2:])

        # 2x, 4x, 8x, 16x, 32x
        features: list[torch.Tensor] = self.foot(x)

        # 32x prediction
        y = self.blocks(features.pop())
        if is_eval:
            y = self.head[0](y)
            y = cropper(y)

        # mutliscale prediction
        # crop before fusion
        y = y + self.neck[-1](x)
        for i in range(4):
            z = self.neck[i](features.pop())
            if is_eval:
                z = self.head[i + 1](z)
                z = cropper(z)

            y = y + z

        return y
