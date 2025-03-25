from io import BufferedReader

from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchlake.common.models import ResBlock
from ..models.darknet53.model import DarkNetBlock
from .convertor_darknet import DarkNetConvertor


class DarkNet53Convertor(DarkNetConvertor):
    def extra_convert(self, weight_handle: BufferedReader, dest: nn.Module):
        if isinstance(dest, ResBlock):
            if not isinstance(dest.downsample, nn.Identity):
                print("start converting shortcut")

                block: Conv2dNormActivation = dest.downsample[0]
                self.convert_bn(weight_handle, block[1])
                self.convert_conv(weight_handle, block[0])

                print("finish converting shortcut")

            if isinstance(dest.block, DarkNetBlock):
                print("start converting darknet 53 block")

                for block in dest.block.blocks:
                    block: Conv2dNormActivation
                    self.convert_bn(weight_handle, block[1])
                    self.convert_conv(weight_handle, block[0])

                print("finish converting darknet 53 block")
