from math import ceil

import pytest
import torch

from ..models.res2net.model import Res2Net
from ..models.res2net.network import BottleNeck, Res2NetLayer


@pytest.mark.parametrize("in_c,split,groups", [[32, 2, 16], [128, 4, 32]])
def test_res2net_layer_forward_shape(in_c: int, split: int, groups: int):
    x = torch.randn(2, in_c, 13, 13)
    layer = Res2NetLayer(in_c, split, groups)
    y = layer(x)

    assert y.shape == torch.Size((2, in_c, 13, 13))


@pytest.mark.parametrize(
    "name,input_channel,base_number,output_channel,stride",
    [
        ["first", 64, 64, 256, 2],
        ["middle", 256, 64, 256, 1],
    ],
)
@pytest.mark.parametrize("split", [2, 4])
@pytest.mark.parametrize("groups", [4, 8])
def test_bottleneck_forward_shape(
    name: str,
    input_channel: int,
    base_number: int,
    output_channel: int,
    stride: int,
    split: int,
    groups: int,
):
    INPUT_SIZE = 13
    OUTPUT_SIZE = ceil(INPUT_SIZE / stride)

    x = torch.randn(2, input_channel, INPUT_SIZE, INPUT_SIZE)
    # 64 -> 128 -> 128 -> 256
    layer = BottleNeck(input_channel, base_number, stride, split, groups)
    y = layer(x)

    assert y.shape == torch.Size((2, output_channel, OUTPUT_SIZE, OUTPUT_SIZE))


@pytest.mark.parametrize("key", [50, 101, 152])
@pytest.mark.parametrize("split", [2, 4])
@pytest.mark.parametrize("groups", [4, 8])
def test_res2net_forward_shape(
    key: int,
    split: int,
    groups: int,
):
    x = torch.randn(2, 3, 224, 224)
    model = Res2Net(
        output_size=5,
        key=key,
        split=split,
        groups=groups,
    )
    y = model(x)

    assert y.shape == torch.Size((2, 5))
