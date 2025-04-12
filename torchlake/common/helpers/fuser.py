from copy import deepcopy
from typing import Callable
import torch
import torch.nn.functional as F

from ..types import BN_TYPE, CONV_TYPE


def pad_conv(dest: CONV_TYPE, target_shape: tuple[int]):
    """pad filter size of dest to filter size of target_conv

    Args:
        conv (CONV_TYPE): convolution layer
        target_conv (CONV_TYPE): convolution layer of target
    """
    # modified weight
    padded_size = []
    for s1, s2 in zip(dest.kernel_size, target_shape):
        padded_size.extend([abs(s2 - s1) // 2] * 2)
    dest.weight.data = F.pad(dest.weight.data, padded_size)

    # modified attribute
    spatial_shape = dest.weight.data.shape[2:]
    dest.kernel_size = spatial_shape
    dest.padding = tuple(s // 2 for s in spatial_shape)


def empty_conv(conv: CONV_TYPE):
    """inplace assign conv's weight and bias to all zeros
    i.e. x -> 0

    Args:
        conv (CONV_TYPE): convolution layer
    """
    conv.weight.data.copy_(torch.zeros_like(conv.weight.data))
    if conv.bias is not None:
        conv.bias.data.copy_(torch.zeros_like(conv.bias.data))


def identity_conv(conv: CONV_TYPE):
    """inplace assign conv's weight and bias to identity transformation
    i.e. x -> x

    Args:
        conv (CONV_TYPE): convolution layer
    """
    conv.weight.data.copy_(torch.ones_like(conv.weight.data))
    if conv.bias is not None:
        conv.bias.data.copy_(torch.zeros_like(conv.bias.data))


def assign_conv(conv: CONV_TYPE, dest: CONV_TYPE):
    """inplace assign conv's weight and bias to destination conv

    Args:
        conv (CONV_TYPE): convolution layer as source
        dest (CONV_TYPE): convolution layer as destination
    """
    dest.weight.data.copy_(conv.weight.data)
    if conv.bias is not None:
        dest.bias.data.copy_(conv.bias.data)


def compensate_numerical_difference(op: Callable, dest: CONV_TYPE):
    input_tensor = torch.randn(1, dest.in_channels, 32, 32)
    with torch.no_grad():
        orig_output = op(input_tensor)
        fused_output = dest(input_tensor)
        diff = orig_output - fused_output
        dest.bias.data += diff.mean(dim=(0, 2, 3))


def fuse_conv_bn(conv: CONV_TYPE, bn: BN_TYPE, dest: CONV_TYPE):
    """fuse conv and bn weight and bias to destination conv
    e.g. conv -> bn

    Args:
        conv (CONV_TYPE): convolution layer as source
        bn (BN_TYPE): batch normalization layer as source
        dest (CONV_TYPE): convolution layer as destination
    """
    # 1 means input channel, last term means spatial dimension
    spatial_dim = len(dest.kernel_size)
    expanded = [1] * (1 + spatial_dim)

    empty_conv(dest)
    assign_conv(conv, dest)
    scale = bn.weight.data / (bn.running_var + bn.eps).sqrt()
    dest.weight.data.mul_(scale.view(-1, *expanded))
    dest.bias.data.sub_(bn.running_mean).mul_(scale).add_(bn.bias.data)


def convert_bn_to_conv(bn: BN_TYPE, dest: CONV_TYPE):
    """convert bn to convolution

    Args:
        bn (BN_TYPE): batch normalization as source
        dest (CONV_TYPE): convolution layer as destination
    """
    empty_conv(dest)
    expanded = [1] * len(dest.kernel_size)

    # modified weights
    scale = bn.weight.data / (bn.running_var + bn.eps).sqrt()
    # make it identity, since bn won't mix channel
    output_dim, input_dim_per_group = dest.weight.data.shape[:2]
    g = dest.groups
    input_dim = input_dim_per_group * g
    assert input_dim == output_dim, "input channel should equal to output channel"

    # out_c, out_c
    # https://github.com/huawei-noah/Efficient-AI-Backbones/blob/f90e129b645c3b1684fe07cd361cd557d0ad71f7/ghostnetv3_pytorch/ghostnetv3.py#L433
    w = torch.diag(scale)
    if g > 1:
        w = torch.cat(
            [
                w[
                    (i * input_dim_per_group) : ((i + 1) * input_dim_per_group),
                    (i * input_dim_per_group) : ((i + 1) * input_dim_per_group),
                ]
                for i in range(g)
            ]
        )
    # out_c, out_c, 1, 1
    w = w.view(output_dim, input_dim_per_group, *expanded)
    # pad kernel size, since bn is 1x1 kernel
    padded_size = []
    for s in dest.kernel_size:
        padded_size.extend([(s - 1) // 2] * 2)
    dest.weight.data.copy_(F.pad(w, padded_size))

    # modified bias
    b = scale.mul(-bn.running_mean).add(bn.bias.data)
    dest.bias.data.copy_(b)


def fuse_sequential_convs(*convs: CONV_TYPE, dest: CONV_TYPE):
    """fuse sequential convolutions
    e.g. 3x3 -> 3x3 -> 3x3

    Args:
        dest (CONV_TYPE): convolution layer as destination
    """
    identity_conv(dest)
    # 1 means input channel, last term means spatial dimension
    spatial_dim = len(dest.kernel_size)
    expanded = [1] * (1 + spatial_dim)

    # w_sort_convs = sorted(list(convs), key=lambda conv: conv.weight.data.size(-1))
    # max_w = w_sort_convs[-1].weight.data.size(-1)
    # for i, conv in enumerate(w_sort_convs[:-1]):
    #     if conv.weight.data.size(-1) < max_w:
    #         pad_conv(w_sort_convs[i], w_sort_convs[-1])

    w = dest.weight.data
    b = dest.bias.data if dest.bias is not None else 1
    for conv in convs:
        w *= conv.weight.data

        if conv.bias is not None:
            b = b.view(-1, *expanded)
            b = b * conv.weight.data
            b += conv.bias.data.view(-1, *expanded)

    if not isinstance(b, torch.Tensor):
        b = 0

    # fusion
    dest.weight.data.copy_(w)
    if dest.bias is not None:
        dest.bias.data.copy_(b)


def fuse_sum_parallel_convs(*convs: CONV_TYPE, dest: CONV_TYPE):
    """fuse sum convolutions
    e.g.   3x3  3x3  3x3
            |    |    |
            v    v    v
            |___(+)___|

    Args:
        dest (CONV_TYPE): convolution layer as destination
    """
    empty_conv(dest)

    w_sort_convs = sorted(list(convs), key=lambda conv: conv.weight.data.size(-1))
    max_w = w_sort_convs[-1].weight.data.size(-1)
    for i, conv in enumerate(w_sort_convs[:-1]):
        if conv.weight.data.size(-1) < max_w:
            pad_conv(w_sort_convs[i], w_sort_convs[-1].kernel_size)

    w = sum(conv.weight.data for conv in convs)
    b = sum(conv.bias.data for conv in convs if conv.bias is not None)

    dest.weight.data.copy_(w)
    if dest.bias is not None:
        dest.bias.data.copy_(b)


def fuse_concat_parallel_convs(*convs: CONV_TYPE, dest: CONV_TYPE):
    """fuse concat convolutions
    e.g.   3x3  3x3   3x3
            |    |     |
            v    v     v
            |__ (||)___|

    Args:
        dest (CONV_TYPE): convolution layer as destination
    """
    empty_conv(dest)

    w_sort_convs = sorted(list(convs), key=lambda conv: conv.weight.data.size(-1))
    max_w = w_sort_convs[-1].weight.data.size(-1)
    for i, conv in enumerate(w_sort_convs[:-1]):
        if conv.weight.data.size(-1) < max_w:
            pad_conv(w_sort_convs[i], w_sort_convs[-1].kernel_size)

    w = torch.cat([conv.weight.data for conv in convs])
    b = [conv.bias.data for conv in convs if conv.bias is not None]
    b = torch.cat(b) if b else 0

    dest.weight.data.copy_(w)
    if dest.bias is not None:
        dest.bias.data.copy_(b)
