from torch import nn


def init_conv_srcnn_style(layer: nn.Conv2d):
    nn.init.normal_(layer.weight, 0, 1e-3)
    nn.init.zeros_(layer.bias)
