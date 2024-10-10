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
from ...mixins.vgg_backbone import DeepLabStyleVGGBackboneMixin


class DeepLab(DeepLabStyleVGGBackboneMixin, nn.Module):

    def __init__(
        self,
        output_size: int = 1,
        large_fov: bool = True,
        frozen_backbone: bool = True,
    ):
        """DeepLab v1 in paper [1412.7062v4]

        Args:
            output_size (int, optional): output size. Defaults to 1.
            large_fov (bool, optional): use larger field of view. Defaults to True.
            fronzen_backbone (bool, optional): froze the vgg backbone or not. Defaults to False.
        """
        super().__init__()
        input_channel = 3
        fc_dim = 1024 if large_fov else 4096

        self.backbone = self.build_backbone("vgg16", frozen_backbone)
        self.conv = nn.Sequential(
            Conv2dNormActivation(
                512,
                fc_dim,
                3 if large_fov else 4,
                padding=12 if large_fov else 6,
                dilation=12 if large_fov else 4,
                norm_layer=None,
            ),
            nn.Dropout(p=0.5),
            Conv2dNormActivation(fc_dim, fc_dim, 1, norm_layer=None),
            nn.Dropout(p=0.5),
            Conv2dNormActivation(fc_dim, output_size, 1, norm_layer=None),
        )
        self.pool_convs = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2dNormActivation(dim, 128, 3, norm_layer=None),
                    nn.Dropout(0.5),
                    Conv2dNormActivation(128, 128, 1, norm_layer=None),
                    nn.Dropout(0.5),
                    nn.Conv2d(128, output_size, 1),
                )
                for dim in [512, 256, 128, 64, input_channel]
            ]
        )

    def train(self: T, mode: bool = True) -> T:
        result = super().train(mode)

        # every feature map downsample to 8x
        for block, s in zip(result.pool_convs, [1, 1, 2, 4, 8]):
            # only modify first encountered conv2dnormactivation
            for layer in block:
                if isinstance(layer, Conv2dNormActivation):
                    layer[0].stride = (s, s)
                    break

        # no upsampling
        # result.upsamples = nn.ModuleList([nn.Identity()] * 5)

        return result

    def eval(self: T) -> T:
        result = super().eval()

        # no downsample for multi scale prediction
        for block in result.pool_convs:
            # only modify first encountered conv2dnormactivation
            for layer in block:
                if isinstance(layer, Conv2dNormActivation):
                    layer[0].stride = (1, 1)
                    break

        # upsample every prediction
        result.upsamples = nn.ModuleList(
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
        # 2x, 4x, 8x, 16x, 32x
        features: list[torch.Tensor] = self.backbone(x, [f"{i}_1" for i in range(1, 6)])
        is_eval = not self.training
        if is_eval:
            cropper = CenterCrop(x.shape[2:])

        # 32x prediction
        y = self.conv(features.pop())
        if is_eval:
            y = self.upsamples[0](y)
            y = cropper(y)

        # mutliscale prediction
        # crop before fusion
        y = y + self.pool_convs[-1](x)
        for i in range(4):
            intermediate = self.pool_convs[i](features.pop())
            if is_eval:
                intermediate = self.upsamples[i + 1](intermediate)
                intermediate = cropper(intermediate)

            y = y + intermediate

        return y
