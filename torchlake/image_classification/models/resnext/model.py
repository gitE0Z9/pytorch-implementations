from ..resnet.model import BottleneckArgs, ConvBlockArgs, ResNet
from .network import BottleNeck, ConvBlock


class ResNeXt(ResNet):
    @property
    def config(self) -> tuple[ConvBlockArgs | BottleneckArgs]:
        """parameter of layers

        Returns:
            tuple[ConvBlockArgs | BottleneckArgs]: parameters of layers
        """
        return (
            self._config
            or {
                18: (
                    ConvBlockArgs(64, 64, 2, ConvBlock),  # less block
                    ConvBlockArgs(64, 128, 2, ConvBlock),  # less block
                    ConvBlockArgs(128, 256, 2, ConvBlock),  # less block
                    ConvBlockArgs(256, 512, 2, ConvBlock),  # less block
                ),
                34: (
                    ConvBlockArgs(64, 64, 3, ConvBlock),  # narrower
                    ConvBlockArgs(64, 128, 4, ConvBlock),  # narrower
                    ConvBlockArgs(128, 256, 6, ConvBlock),  # narrower
                    ConvBlockArgs(256, 512, 3, ConvBlock),  # narrower
                ),
                50: (
                    BottleneckArgs(64, 128, 2, 3, BottleNeck),
                    BottleneckArgs(256, 256, 2, 4, BottleNeck),
                    BottleneckArgs(512, 512, 2, 6, BottleNeck),
                    BottleneckArgs(1024, 1024, 2, 3, BottleNeck),
                ),
                101: (
                    BottleneckArgs(64, 128, 2, 3, BottleNeck),
                    BottleneckArgs(256, 256, 2, 4, BottleNeck),
                    BottleneckArgs(512, 512, 2, 23, BottleNeck),  # more block
                    BottleneckArgs(1024, 1024, 2, 3, BottleNeck),
                ),
                152: (
                    BottleneckArgs(64, 128, 2, 3, BottleNeck),
                    BottleneckArgs(256, 256, 2, 8, BottleNeck),  # more block
                    BottleneckArgs(512, 512, 2, 36, BottleNeck),  # more block
                    BottleneckArgs(1024, 1024, 2, 3, BottleNeck),
                ),
            }[self.key]
        )
