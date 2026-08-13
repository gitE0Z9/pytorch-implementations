import pytest
import torch

from ..models.tcn.model import TCN
from ..models.tcn.network import BottleNeck, CausalConv1d

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
SEQ_LEN = 1024
OUTPUT_SIZE = 10


class TestNetwork:
    def test_causal_conv_1d_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, SEQ_LEN))
        model = CausalConv1d(HIDDEN_DIM, HIDDEN_DIM + 2, 2, padding=1)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM + 2, SEQ_LEN))

    def test_bottleneck_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, SEQ_LEN))
        model = BottleNeck(HIDDEN_DIM, HIDDEN_DIM + 2, 2, padding=1)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM + 2, SEQ_LEN))


class TestModel:
    @pytest.mark.parametrize(
        "output_sequence,expected",
        (
            (True, (BATCH_SIZE, OUTPUT_SIZE, SEQ_LEN)),
            (False, (BATCH_SIZE, OUTPUT_SIZE)),
        ),
    )
    def test_tcn_forward_shape(self, output_sequence: bool, expected: tuple[int]):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, SEQ_LEN))
        model = TCN(
            INPUT_CHANNEL,
            [32, 64, 128],
            OUTPUT_SIZE,
            num_block=3,
            output_sequence=output_sequence,
        )
        y = model(x)

        assert y.shape == torch.Size(expected)
