import re
import torch
import torch.nn as nn
import numpy as np
import os
from models.yolov1.network import Extraction
from models.yolov2.network import Darknet19, ConvBlock, BottleNeck


def parse_config(cfg_path: str):
    """Buggy."""

    output = []

    with open(cfg_path, "r") as f:
        text = f.read()

    titles = re.findall("\[(.*?)\]", text)

    present_pos = 0
    next_pos = 0

    for index, present_block in enumerate(titles):
        if index == len(titles) - 1:
            break

        present_pos = text.index(present_block, next_pos)
        next_pos = text.index(titles[index + 1], present_pos)

        parameters = re.findall("(.*?)=(.*?)\n", text[present_pos:next_pos])
        parameters = {k: v for k, v in parameters}

        output.append({present_block: parameters})

    return output


def convert_weight(model_name: str, weight_path: str):
    assert model_name in ["extraction", "darknet19"]

    if model_name == "darknet19":
        model = Darknet19()
    elif model_name == "extraction":
        model = Extraction()
    else:
        raise NotImplementedError

    weight_handle = open(weight_path, "rb")
    _ = np.fromfile(weight_handle, count=4, dtype=np.int32)

    for network_module in model.children():
        if isinstance(network_module, nn.Sequential):
            for layer in network_module:
                if isinstance(layer, ConvBlock):
                    bb = np.fromfile(
                        weight_handle,
                        count=layer.bn.bias.numel(),
                        dtype=np.float32,
                    )
                    bw = np.fromfile(
                        weight_handle,
                        count=layer.bn.weight.numel(),
                        dtype=np.float32,
                    )
                    bm = np.fromfile(
                        weight_handle,
                        count=layer.bn.running_mean.numel(),
                        dtype=np.float32,
                    )
                    bv = np.fromfile(
                        weight_handle,
                        count=layer.bn.running_var.numel(),
                        dtype=np.float32,
                    )
                    cw = np.fromfile(
                        weight_handle,
                        count=layer.conv.weight.numel(),
                        dtype=np.float32,
                    )
                    layer.bn.bias.data.copy_(torch.from_numpy(bb))
                    layer.bn.weight.data.copy_(torch.from_numpy(bw))
                    layer.bn.running_mean.data.copy_(torch.from_numpy(bm))
                    layer.bn.running_var.data.copy_(torch.from_numpy(bv))
                    layer.conv.weight.data.copy_(
                        torch.from_numpy(cw).reshape(layer.conv.weight.shape)
                    )
                    print("a")
                elif isinstance(layer, BottleNeck):
                    for g in layer.conv:
                        if isinstance(g, ConvBlock):
                            bb = np.fromfile(
                                weight_handle,
                                count=g.bn.bias.numel(),
                                dtype=np.float32,
                            )
                            bw = np.fromfile(
                                weight_handle,
                                count=g.bn.weight.numel(),
                                dtype=np.float32,
                            )
                            bm = np.fromfile(
                                weight_handle,
                                count=g.bn.running_mean.numel(),
                                dtype=np.float32,
                            )
                            bv = np.fromfile(
                                weight_handle,
                                count=g.bn.running_var.numel(),
                                dtype=np.float32,
                            )
                            cw = np.fromfile(
                                weight_handle,
                                count=g.conv.weight.numel(),
                                dtype=np.float32,
                            )
                            g.bn.bias.data.copy_(torch.from_numpy(bb))
                            g.bn.weight.data.copy_(torch.from_numpy(bw))
                            g.bn.running_mean.data.copy_(torch.from_numpy(bm))
                            g.bn.running_var.data.copy_(torch.from_numpy(bv))
                            g.conv.weight.data.copy_(
                                torch.from_numpy(cw).reshape(g.conv.weight.shape)
                            )
                            print("b")
                elif isinstance(layer, nn.Conv2d):
                    b = np.fromfile(
                        weight_handle,
                        count=layer.bias.numel(),
                        dtype=np.float32,
                    )
                    w = np.fromfile(
                        weight_handle,
                        count=layer.weight.numel(),
                        dtype=np.float32,
                    )
                    layer.bias.data.copy_(torch.from_numpy(b))
                    layer.weight.data.copy_(
                        torch.from_numpy(w).reshape(layer.weight.shape)
                    )
                    print("c")

    print("remaning:", np.fromfile(weight_handle, dtype=np.float32).shape)

    weight_handle.close()

    output_weight_path = os.path.basename(weight_path).replace("weights", "pth")
    torch.save(model.state_dict(), output_weight_path)
