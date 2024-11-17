import os
import re
from io import BufferedReader

import numpy as np
import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class DarkNetConvertor:
    # def parse_config(cfg_path: str):
    #     """Buggy."""

    #     output = []

    #     with open(cfg_path, "r") as f:
    #         text = f.read()

    #     titles = re.findall("\[(.*?)\]", text)

    #     present_pos = 0
    #     next_pos = 0

    #     for index, present_block in enumerate(titles):
    #         if index == len(titles) - 1:
    #             break

    #         present_pos = text.index(present_block, next_pos)
    #         next_pos = text.index(titles[index + 1], present_pos)

    #         parameters = re.findall("(.*?)=(.*?)\n", text[present_pos:next_pos])
    #         parameters = {k: v for k, v in parameters}

    #         output.append({present_block: parameters})

    #     return output

    def convert_bn(self, weight_handle: BufferedReader, dest: nn.BatchNorm2d):
        print("start converting bn")

        bb = np.fromfile(
            weight_handle,
            count=dest.bias.numel(),
            dtype=np.float32,
        )
        bw = np.fromfile(
            weight_handle,
            count=dest.weight.numel(),
            dtype=np.float32,
        )
        bm = np.fromfile(
            weight_handle,
            count=dest.running_mean.numel(),
            dtype=np.float32,
        )
        bv = np.fromfile(
            weight_handle,
            count=dest.running_var.numel(),
            dtype=np.float32,
        )

        dest.bias.data.copy_(torch.from_numpy(bb))
        dest.weight.data.copy_(torch.from_numpy(bw))
        dest.running_mean.data.copy_(torch.from_numpy(bm))
        dest.running_var.data.copy_(torch.from_numpy(bv))

        print("finish converting bn")

    def convert_conv(self, weight_handle: BufferedReader, dest: nn.Conv2d):
        print("start converting conv")

        if dest.bias is not None:
            cb = np.fromfile(
                weight_handle,
                count=dest.bias.numel(),
                dtype=np.float32,
            )

        cw = np.fromfile(
            weight_handle,
            count=dest.weight.numel(),
            dtype=np.float32,
        )

        dest.weight.data.copy_(torch.from_numpy(cw).reshape(dest.weight.shape))

        if dest.bias is not None:
            dest.bias.data.copy_(torch.from_numpy(cb).reshape(dest.bias.shape))

        print("finish converting conv")

    def extra_convert(self, weight_handle: BufferedReader, dest: nn.Module):
        pass

    def run(self, weights_path: str, dest: nn.Module):
        weight_handle = open(weights_path, "rb")
        _ = np.fromfile(weight_handle, count=4, dtype=np.int32)

        for network_module in dest.children():
            if isinstance(network_module, nn.Sequential):
                for layer in network_module:
                    if isinstance(layer, Conv2dNormActivation):
                        bn_layer = None
                        if isinstance(layer[1], nn.BatchNorm2d):
                            bn_layer = layer[1]

                            self.convert_bn(weight_handle, bn_layer)

                        self.convert_conv(weight_handle, layer[0])

                        print("Conv2dNormActivation")

                    elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        self.convert_conv(weight_handle, layer)

                        print("conv / linear")

                    else:
                        self.extra_convert(weight_handle, layer)

        remaining_bits = np.fromfile(weight_handle, dtype=np.float32).shape[0]

        print("remaning:", remaining_bits)

        weight_handle.close()

        return remaining_bits
