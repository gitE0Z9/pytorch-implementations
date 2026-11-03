from torch import nn
from ..resnet.model import ResNet
from .network import BottleNeck, ConvBlock


class ResNeXt(ResNet):
    @property
    def config(self) -> tuple[tuple[int | nn.Module]]:
        """parameter of layers

        Returns:
            list[list[int | nn.Module]]: parameters of layers
        """
        return {
            # input_dim, output_dim, number_block, block_cls
            18: (
                (64, 64, 2, ConvBlock),  # less block
                (64, 128, 2, ConvBlock),  # less block
                (128, 256, 2, ConvBlock),  # less block
                (256, 512, 2, ConvBlock),  # less block
            ),
            34: (
                (64, 64, 3, ConvBlock),  # narrower
                (64, 128, 4, ConvBlock),  # narrower
                (128, 256, 6, ConvBlock),  # narrower
                (256, 512, 3, ConvBlock),  # narrower
            ),
            # input_dim, hidden_dim, expansion, number_block, block_cls
            50: (
                (64, 64, 4, 3, BottleNeck),
                (256, 128, 4, 4, BottleNeck),
                (512, 256, 4, 6, BottleNeck),
                (1024, 512, 4, 3, BottleNeck),
            ),
            101: (
                (64, 64, 4, 3, BottleNeck),
                (256, 128, 4, 4, BottleNeck),
                (512, 256, 4, 23, BottleNeck),  # more block
                (1024, 512, 4, 3, BottleNeck),
            ),
            152: (
                (64, 64, 4, 3, BottleNeck),
                (256, 128, 4, 8, BottleNeck),  # more block
                (512, 256, 4, 36, BottleNeck),  # more block
                (1024, 512, 4, 3, BottleNeck),
            ),
        }[self.key]
