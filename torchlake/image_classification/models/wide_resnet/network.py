from torch import nn

from ..resnet.network import ConvBlock


class DropoutConvBlock(ConvBlock):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        stride: int = 1,
        pre_activation: bool = False,
        dropout_prob: float = 0.5,
    ):
        """convolution block in resnet
        kernel: 3x3 -> 3x3
        channel: input_channel -> output_channel -> output_channel

        Args:
            input_channel (int): input channel size
            output_channel (int): output channel size
            stride (int, optional): stride of block. Defaults to 1.
            pre_activation (bool, optional): put activation before convolution layer in paper[1603.05027v3]. Defaults to False.
            dropout_prob (float, optional): enable dropout layer between layers. Defaults to 0.5.
        """
        super().__init__(
            input_channel,
            output_channel,
            stride=stride,
            pre_activation=pre_activation,
        )
        if dropout_prob > 0:
            self.layers.insert(1, nn.Dropout(dropout_prob))
