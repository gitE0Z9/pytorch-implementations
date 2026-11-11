from ..resnet.model import BottleneckArgs, ResNet
from .network import BottleNeck


class SkNet(ResNet):
    @property
    def config(self) -> tuple[BottleneckArgs]:
        """parameter of layers

        Returns:
            tuple[BottleneckArgs]: parameters of layers
        """
        return (
            self._config
            or {
                26: (
                    BottleneckArgs(64, 128, 2, 2, BottleNeck),  # less block
                    BottleneckArgs(256, 256, 2, 2, BottleNeck),  # less block
                    BottleneckArgs(512, 512, 2, 2, BottleNeck),  # less block
                    BottleneckArgs(1024, 1024, 2, 2, BottleNeck),  # less block
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
