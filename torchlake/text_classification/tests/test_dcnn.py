import pytest
import torch

from ..models.dcnn import DCNN, DynamicKmaxPool1d
from ..models.dcnn.network import Block, Folding, WideConv1d

BATCH_SIZE = 4
MAX_SEQ_LEN = 256
VOCAB_SIZE = 10


class TestNetwork:
    def test_wide_conv_forward_shape(self):
        model = WideConv1d(2, 1, 5)

        x = torch.rand((BATCH_SIZE, 2, 7))
        output = model(x)

        assert output.shape == torch.Size((BATCH_SIZE, 1, 11))

    @pytest.mark.parametrize("conv_layer_idx,expected", [[1, 12], [2, 6], [3, 3]])
    def test_dynamic_max_pool_forward_shape(self, conv_layer_idx: int, expected: int):
        model = DynamicKmaxPool1d(3, 18, conv_layer_idx, 3)

        x = torch.rand((BATCH_SIZE, 1, 18))
        output = model(x)

        assert output.shape == torch.Size((BATCH_SIZE, 1, expected))

    @pytest.mark.parametrize("conv_layer_idx,expected", [[1, 12], [2, 6], [3, 3]])
    def test_block_forward_shape(self, conv_layer_idx: int, expected: int):
        model = Block(1, 1, 5, 3, 18, conv_layer_idx, 3)

        x = torch.rand((BATCH_SIZE, 1, 18))
        output = model(x)

        assert output.shape == torch.Size((BATCH_SIZE, 1, expected))

    def test_folding_forward_shape(self):
        model = Folding()

        x = torch.rand((BATCH_SIZE, 8, MAX_SEQ_LEN))
        output = model(x)

        assert output.shape == torch.Size((BATCH_SIZE, 4, MAX_SEQ_LEN))


class TestModel:
    def test_dcnn_forward_shape(self):
        model = DCNN(VOCAB_SIZE, topk=3)

        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
        output = model.forward(x)

        assert output.shape == torch.Size((BATCH_SIZE, 1))
