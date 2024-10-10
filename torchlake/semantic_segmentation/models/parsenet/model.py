# for further speed up, split import is promising
from annotated_types import T
from torch import nn
from torchvision.ops import Conv2dNormActivation

from ..deeplab import DeepLab
from ..fcn import FCN
from .network import GlobalContextModule


class ParseNetFCN(FCN):

    def __init__(
        self,
        output_size: int = 1,
        num_skip_connection: int = 2,
        frozen_backbone: bool = False,
    ):
        """ParseNet [1506.04579v2] with Fully Convolutional Networks for Semantic Segmentation in paper [1605.06211v1]

        Args:
            output_size (int, optional): output size. Defaults to 1.
            num_skip_connection (int, optional): number of skip connection, e.g. 2 means FCN-8s, 1 means FCN-16s, 0 means FCN-32s. Defaults to 2.
            fronzen_backbone (bool, optional): froze the vgg backbone or not. Defaults to False.
        """
        super().__init__(output_size, num_skip_connection, frozen_backbone)
        self.conv = nn.Sequential(
            Conv2dNormActivation(512, 1024, 3, padding=3, dilation=3, norm_layer=None),
            nn.Dropout(p=0.5),
            Conv2dNormActivation(1024, 1024, 1, norm_layer=None),
            nn.Dropout(p=0.5),
            nn.Conv2d(1024, output_size, 1),
        )
        self.pool_convs = nn.ModuleList(
            [
                nn.Sequential(
                    GlobalContextModule(dim),
                    nn.Conv2d(2 * dim, output_size, 1),
                )
                for dim in [512, 256, 128, 64][:num_skip_connection]
            ]
        )


class ParseNetDeepLab(DeepLab):

    def __init__(
        self,
        output_size: int = 1,
        large_fov: bool = True,
        frozen_backbone: bool = True,
    ):
        """ParseNet [1506.04579v2] with DeepLab v1 in paper [1412.7062v4]

        Args:
            output_size (int, optional): output size. Defaults to 1.
            large_fov (bool, optional): use larger field of view. Defaults to True.
            fronzen_backbone (bool, optional): froze the vgg backbone or not. Defaults to False.
        """
        super().__init__(output_size, large_fov, frozen_backbone)
        input_channel = 3

        self.pool_convs = nn.ModuleList(
            [
                nn.Sequential(
                    GlobalContextModule(dim),
                    Conv2dNormActivation(2 * dim, 128, 3, norm_layer=None),
                    nn.Dropout(0.5),
                    Conv2dNormActivation(128, 128, 1, norm_layer=None),
                    nn.Dropout(0.5),
                    nn.Conv2d(128, output_size, 1),
                )
                for dim in [512, 256, 128, 64, input_channel]
            ]
        )
