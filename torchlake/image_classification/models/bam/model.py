from torch import nn

from ..resnet.model import ResNet
from ..resnet.network import ResBlock
from .network import BAM2d


class BAMResNet(ResNet):

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
                        BAM2d(output_channel),
                        nn.MaxPool2d(2, 2),
                    ]
                )

            setattr(self, f"block{block_index+1}", nn.Sequential(*layers))
