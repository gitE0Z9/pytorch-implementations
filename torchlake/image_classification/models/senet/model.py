from ..resnet.model import BottleneckArgs, ConvBlockArgs, ResNet
from .network import SEBottleNeck, SEConvBlock, SEXBottleNeck, SEXConvBlock


class SEResNet(ResNet):
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
                    ConvBlockArgs(64, 64, 2, SEConvBlock),  # less block
                    ConvBlockArgs(64, 128, 2, SEConvBlock),  # less block
                    ConvBlockArgs(128, 256, 2, SEConvBlock),  # less block
                    ConvBlockArgs(256, 512, 2, SEConvBlock),  # less block
                ),
                34: (
                    ConvBlockArgs(64, 64, 3, SEConvBlock),  # narrower
                    ConvBlockArgs(64, 128, 4, SEConvBlock),  # narrower
                    ConvBlockArgs(128, 256, 6, SEConvBlock),  # narrower
                    ConvBlockArgs(256, 512, 3, SEConvBlock),  # narrower
                ),
                50: (
                    BottleneckArgs(64, 64, 4, 3, SEBottleNeck),
                    BottleneckArgs(256, 128, 4, 4, SEBottleNeck),
                    BottleneckArgs(512, 256, 4, 6, SEBottleNeck),
                    BottleneckArgs(1024, 512, 4, 3, SEBottleNeck),
                ),
                101: (
                    BottleneckArgs(64, 64, 4, 3, SEBottleNeck),
                    BottleneckArgs(256, 128, 4, 4, SEBottleNeck),
                    BottleneckArgs(512, 256, 4, 23, SEBottleNeck),  # more block
                    BottleneckArgs(1024, 512, 4, 3, SEBottleNeck),
                ),
                152: (
                    BottleneckArgs(64, 64, 4, 3, SEBottleNeck),
                    BottleneckArgs(256, 128, 4, 8, SEBottleNeck),  # more block
                    BottleneckArgs(512, 256, 4, 36, SEBottleNeck),  # more block
                    BottleneckArgs(1024, 512, 4, 3, SEBottleNeck),
                ),
            }[self.key]
        )


class SEResNeXt(ResNet):
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
                    ConvBlockArgs(64, 64, 2, SEXConvBlock),  # less block
                    ConvBlockArgs(64, 128, 2, SEXConvBlock),  # less block
                    ConvBlockArgs(128, 256, 2, SEXConvBlock),  # less block
                    ConvBlockArgs(256, 512, 2, SEXConvBlock),  # less block
                ),
                34: (
                    ConvBlockArgs(64, 64, 3, SEXConvBlock),  # narrower
                    ConvBlockArgs(64, 128, 4, SEXConvBlock),  # narrower
                    ConvBlockArgs(128, 256, 6, SEXConvBlock),  # narrower
                    ConvBlockArgs(256, 512, 3, SEXConvBlock),  # narrower
                ),
                50: (
                    BottleneckArgs(64, 128, 2, 3, SEXBottleNeck),
                    BottleneckArgs(256, 256, 2, 4, SEXBottleNeck),
                    BottleneckArgs(512, 512, 2, 6, SEXBottleNeck),
                    BottleneckArgs(1024, 1024, 2, 3, SEXBottleNeck),
                ),
                101: (
                    BottleneckArgs(64, 128, 2, 3, SEXBottleNeck),
                    BottleneckArgs(256, 256, 2, 4, SEXBottleNeck),
                    BottleneckArgs(512, 512, 2, 23, SEXBottleNeck),  # more block
                    BottleneckArgs(1024, 1024, 2, 3, SEXBottleNeck),
                ),
                152: (
                    BottleneckArgs(64, 128, 2, 3, SEXBottleNeck),
                    BottleneckArgs(256, 256, 2, 8, SEXBottleNeck),  # more block
                    BottleneckArgs(512, 512, 2, 36, SEXBottleNeck),  # more block
                    BottleneckArgs(1024, 1024, 2, 3, SEXBottleNeck),
                ),
            }[self.key]
        )
