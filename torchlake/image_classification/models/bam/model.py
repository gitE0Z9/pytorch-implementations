from typing import Any

from torch import nn
from torchlake.common.models import Bam2d

from ..resnet.model import CONFIGS, ResNet
from ..resnet.network import ResBlock


class BamResNet(ResNet):
    def __init__(
        self,
        input_channel: int = 3,
        output_size: int = 1,
        num_layer: int = 50,
        pre_activation: bool = False,
        configs: dict[int, Any] = CONFIGS,
    ):
        super(BamResNet, self).__init__(
            input_channel,
            output_size,
            num_layer,
            pre_activation,
            configs,
        )

    def build_blocks(self):
        for block_index, (
            input_channel,
            output_channel,
            base_number,
            num_layer,
            layer_class,
        ) in enumerate(self.config):
            layers = [
                ResBlock(
                    input_channel if layer_index == 0 else output_channel,
                    base_number,
                    output_channel,
                    layer_class,
                    pre_activation=self.pre_activation,
                )
                for layer_index in range(num_layer)
            ]
            if block_index not in [0, len(self.config) - 1]:
                layers.extend(
                    [
                        Bam2d(output_channel),
                        nn.MaxPool2d(2, 2),
                    ]
                )

            setattr(self, f"block{block_index+1}", nn.Sequential(*layers))