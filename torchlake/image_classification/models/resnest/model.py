from torch import nn

from ..resnet.model import BottleneckArgs, ResNet
from .network import BottleNeck


class ResNeSt(ResNet):

    @property
    def config(self) -> tuple[BottleneckArgs]:
        """parameter of layers

        Returns:
            tuple[BottleneckArgs]: parameters of layers
        """
        return (
            self._config
            or {
                50: (
                    BottleneckArgs(64, 64, 4, 3, BottleNeck),
                    BottleneckArgs(256, 128, 4, 4, BottleNeck),
                    BottleneckArgs(512, 256, 4, 6, BottleNeck),
                    BottleneckArgs(1024, 512, 4, 3, BottleNeck),
                ),
                101: (
                    BottleneckArgs(64, 64, 4, 3, BottleNeck),
                    BottleneckArgs(256, 128, 4, 4, BottleNeck),
                    BottleneckArgs(512, 256, 4, 23, BottleNeck),  # more block
                    BottleneckArgs(1024, 512, 4, 3, BottleNeck),
                ),
                152: (
                    BottleneckArgs(64, 64, 4, 3, BottleNeck),
                    BottleneckArgs(256, 128, 4, 8, BottleNeck),  # more block
                    BottleneckArgs(512, 256, 4, 36, BottleNeck),  # more block
                    BottleneckArgs(1024, 512, 4, 3, BottleNeck),
                ),
            }[self.key]
        )
