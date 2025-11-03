from torch import nn


def init_conv_dcgan_style(layer: nn.Conv2d):
    nn.init.normal_(layer.weight, 0, 0.02)


def init_bn_dcgan_style(layer: nn.BatchNorm2d):
    nn.init.normal_(layer.weight, 1, 0.02)
    nn.init.constant_(layer.bias, 0)


def init_conv_ebgan_style(layer: nn.Conv2d):
    nn.init.normal_(layer.weight, 0, 0.002)
