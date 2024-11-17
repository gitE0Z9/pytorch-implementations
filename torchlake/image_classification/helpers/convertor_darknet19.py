from io import BufferedReader

from torch import nn
from torchvision.ops import Conv2dNormActivation

from ..models.darknet19.model import BottleNeck
from .convertor_darknet import DarkNetConvertor


class DarkNet19Convertor(DarkNetConvertor):
    def extra_convert(self, weight_handle: BufferedReader, dest: nn.Module):
        if isinstance(dest, BottleNeck):
            for block in dest.blocks:
                print("start converting bottleneck")

                block: Conv2dNormActivation
                self.convert_bn(weight_handle, block[1])
                self.convert_conv(weight_handle, block[0])

                print("finish converting bottleneck")
