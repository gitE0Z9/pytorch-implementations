from torch import nn

from ..ghostnetv2.model import GhostNetV2
from .network import GhostLayerV3


class GhostNetV3(GhostNetV2):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        width_multiplier: float = 1,
    ):
        """GhostNet version 3 [2404.11202v1]

        Args:
            input_channel (int): input channel size. Defaults to 3.
            output_size (int, optional): output size. Defaults to 1.
            width_multiplier (float, optional): width multiplier alpha. Defaults to 1.
        """
        self.width_multiplier = width_multiplier
        super().__init__(input_channel, output_size)

    def build_blocks(self):
        self.blocks = nn.Sequential(
            *[
                GhostLayerV3(
                    int(in_c * self.width_multiplier),
                    int(out_c * self.width_multiplier),
                    kernel,
                    stride=stride,
                    s=2,
                    d=3,
                    expansion_size=expansion_size,
                    enable_se=enable_se,
                    horizontal_kernel=5,
                    vertical_kernel=5,
                )
                for in_c, out_c, kernel, stride, expansion_size, enable_se in self.config
            ]
        )

    def reparameterize(self, target: GhostNetV2):
        for src, dest in zip(self.blocks, target.blocks):
            src.reparameterize(dest)

        target.foot.load_state_dict(self.foot.state_dict())
        target.neck.load_state_dict(self.neck.state_dict())
        target.head.load_state_dict(self.head.state_dict())
