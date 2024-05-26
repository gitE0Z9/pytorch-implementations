from torchlake.common.mixins.network import SeMixin
from torchlake.common.models import CoordinateAttention2d

from ..resnet.network import BottleNeck


class BottleNeck(SeMixin, BottleNeck):
    def __init__(
        self,
        input_channel: int,
        block_base_channel: int,
        pre_activation: bool = False,
    ):
        """bottleneck block in coordinate attention resnet
        1 -> 3 -> 1
        input_channel -> block_base_channel -> block_base_channel -> 4 * block_base_channel

        Args:
            input_channel (int): input channel size
            block_base_channel (int): base number of block channel size
            pre_activation (bool, Defaults False): put activation before transformation [1603.05027v3]
        """
        super(BottleNeck, self).__init__(
            input_channel,
            block_base_channel,
            pre_activation,
        )
        self.se = CoordinateAttention2d(block_base_channel * 4, 32)
