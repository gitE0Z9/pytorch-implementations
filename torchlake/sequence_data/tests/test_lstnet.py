import pytest
import torch

from ..models.lstnet.model import LSTNet
from ..models.lstnet.network import Highway, SkipRNN

BATCH_SIZE = 2
HIDDEN_DIM = 100
SEQ_LEN = 24 * 30
OUTPUT_SIZE = 5

WINDOW_SIZE = 24 * 7
KERNEL = 6


class TestNetwork:
    def test_forward_shape_skiprnn(self):
        c = torch.rand((BATCH_SIZE, HIDDEN_DIM, SEQ_LEN - KERNEL + 1))
        r = torch.rand((BATCH_SIZE, HIDDEN_DIM))

        model = SkipRNN(HIDDEN_DIM, 5, KERNEL, WINDOW_SIZE, 24, 0.2)

        y = model(c, r)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM + 5 * 24))

    def test_forward_shape_highway(self):
        x = torch.rand(BATCH_SIZE, 1, WINDOW_SIZE, OUTPUT_SIZE)
        z = torch.rand(BATCH_SIZE, OUTPUT_SIZE)

        model = Highway(24)

        y = model(x, z)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))


class TestModel:
    @pytest.mark.parametrize("highway_window_size", [0, 24])
    @pytest.mark.parametrize("skip_window_size", [0, 24])
    def test_forward_shape(
        self,
        highway_window_size: int,
        skip_window_size: int,
    ):
        x = torch.rand(BATCH_SIZE, 1, SEQ_LEN, OUTPUT_SIZE)

        model = LSTNet(
            output_size=OUTPUT_SIZE,
            highway_window_size=highway_window_size,
            skip_window_size=skip_window_size,
        )

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))
