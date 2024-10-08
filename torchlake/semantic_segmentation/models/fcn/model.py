import torch
from torch import nn
from torchlake.common.models import VGGFeatureExtractor
from torchvision.ops import Conv2dNormActivation
from torchvision.transforms import CenterCrop

from .network import UpSampling, UpSamplingLegacy


class FCNLegacy(nn.Module):
    def __init__(
        self,
        output_size: int = 1,
        num_skip_connection: int = 2,
        frozen_backbone: bool = False,
    ):
        """Fully Convolutional Networks for Semantic Segmentation in paper [1411.4038v2]

        Args:
            output_size (int, optional): output size. Defaults to 1.
            num_skip_connection (int, optional): number of skip connection, e.g. 2 means FCN-8s, 1 means FCN-16s, 0 means FCN-32s. Defaults to 2.
            fronzen_backbone (bool, optional): froze the vgg backbone or not. Defaults to False.
        """
        super().__init__()
        self.num_skip_connection = num_skip_connection
        self.backbone = VGGFeatureExtractor(
            "vgg16",
            "maxpool",
            trainable=not frozen_backbone,
        )
        self.upsamples = nn.ModuleList(
            [
                # intermediate upsample
                *[
                    UpSamplingLegacy(in_c, out_c, 2, stride=2)
                    for in_c, out_c in (
                        # (512, 512),  # pool5
                        (512, 512),  # pool5 -> pool4 for 16s
                        (512, 256),  # pool4 -> pool3
                        (256, 128),  # pool3 -> pool2
                        (128, 64),  # pool2 -> pool1
                    )[:num_skip_connection]
                ],
                # final upsample
                UpSamplingLegacy(
                    [512, 512, 256][num_skip_connection],  # 32s, 16s, 8s
                    output_size,
                    # e.g. 3 => 8s => 8, 2 => 16s => 16
                    kernel=2 ** (5 - num_skip_connection),
                    # e.g. 3 => 8s => 8, 2 => 16s => 16
                    stride=2 ** (5 - num_skip_connection),
                ),
            ]
        )
        self.fc = nn.Conv2d(output_size, output_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers_no = [f"{i}_1" for i in range(5, 4 - self.num_skip_connection, -1)]
        features = self.backbone(x, layers_no)

        y = features.pop()  # 32x, 512
        for upsample in self.upsamples[:-1]:
            y = upsample(y) + features.pop()  # 16x, 256 # 8x, 128

        y = self.upsamples[-1](y)
        return self.fc(y)


class FCN(nn.Module):
    def __init__(
        self,
        output_size: int = 1,
        num_skip_connection: int = 2,
        frozen_backbone: bool = False,
    ):
        """Fully Convolutional Networks for Semantic Segmentation in paper [1605.06211v1]

        Args:
            output_size (int, optional): output size. Defaults to 1.
            num_skip_connection (int, optional): number of skip connection, e.g. 2 means FCN-8s, 1 means FCN-16s, 0 means FCN-32s. Defaults to 2.
            fronzen_backbone (bool, optional): froze the vgg backbone or not. Defaults to False.
        """
        super().__init__()
        self.num_skip_connection = num_skip_connection
        self.backbone = self.build_backbone(frozen_backbone)
        self.conv = nn.Sequential(
            Conv2dNormActivation(512, 4096, 7, padding=0, norm_layer=None),
            nn.Dropout(p=0.5),
            Conv2dNormActivation(4096, 4096, 1, padding=0, norm_layer=None),
            nn.Dropout(p=0.5),
            nn.Conv2d(4096, output_size, 1),
        )
        self.pool_convs = nn.ModuleList(
            [
                nn.Conv2d(dim, output_size, 1)
                for dim in [512, 256, 128, 64][:num_skip_connection]
            ]
        )
        # intermediate upsample
        self.upsamples = nn.ModuleList(
            [UpSampling(output_size, output_size, 4, stride=2)] * num_skip_connection
        )
        # final upsample
        self.upsample_head = UpSampling(
            output_size,
            output_size,
            # e.g. 2 => 8s => 16, 1 => 16s => 32
            kernel=2 ** (6 - num_skip_connection),
            # e.g. 2 => 8s => 8, 1 => 16s => 16
            stride=2 ** (5 - num_skip_connection),
        )

    def build_backbone(self, frozen_backbone: bool) -> VGGFeatureExtractor:
        backbone = VGGFeatureExtractor(
            "vgg16",
            "maxpool",
            trainable=not frozen_backbone,
        )
        feature_layers = backbone.feature_extractor
        feature_layers[0].padding = (100, 100)

        return backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 8s => 5, 4, 3
        # 16s => 5, 4
        # 32s => 5
        layers_no = [f"{i}_1" for i in range(5, 4 - self.num_skip_connection, -1)]
        features = self.backbone(x, layers_no)

        # feature fusion
        y = self.conv(features.pop())  # 32x
        for upsample, pool_conv in zip(self.upsamples, self.pool_convs):
            y = upsample(y)
            y += CenterCrop(y.shape[-2:])(pool_conv(features.pop()))

        y = self.upsample_head(y)
        return CenterCrop(x.shape[-2:])(y)
