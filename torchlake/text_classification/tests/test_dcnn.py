import pytest
import torch

from ..models.dcnn import DCNN, DynamicKmaxPool1d
from ..models.dcnn.network import Block, Folding, WideConv1d


def test_wide_conv_output_shape():
    model = WideConv1d(1, 1, 5)

    x = torch.rand((4, 1, 7))
    output = model.forward(x)

    assert output.shape == torch.Size((4, 1, 11))


@pytest.mark.parametrize("conv_layer_idx,expected", [[1, 12], [2, 6], [3, 3]])
def test_dynamic_max_pool_output_shape(conv_layer_idx: int, expected: int):
    model = DynamicKmaxPool1d(3, 18, conv_layer_idx, 3)

    x = torch.rand((4, 1, 18))
    output = model.forward(x)

    assert output.shape == torch.Size((4, 1, expected))


@pytest.mark.parametrize("conv_layer_idx,expected", [[1, 12], [2, 6], [3, 3]])
def test_block_output_shape(conv_layer_idx: int, expected: int):
    model = Block(1, 1, 5, 3, 18, conv_layer_idx, 3)

    x = torch.rand((4, 1, 18))
    output = model.forward(x)

    assert output.shape == torch.Size((4, 1, expected))


def test_folding_output_shape():
    model = Folding()

    x = torch.rand((4, 8, 16))
    output = model.forward(x)

    assert output.shape == torch.Size((4, 4, 16))


def test_dcnn_output_shape():
    model = DCNN(10, topk=3)

    x = torch.randint(0, 10, (4, 256))
    output = model.forward(x)

    assert output.shape == torch.Size((4, 1))
