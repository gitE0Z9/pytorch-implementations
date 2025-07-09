import pytest
import torch

from ..models.pixelrnn.model import PixelRNN, MultiScalePixelRNN
from ..models.pixelrnn.network import (
    BottleNeck,
    RowLSTM,
    DiagonalLSTMCell,
    DiagonalLSTM,
)

BATCH_SIZE = 2
IMAGE_SIZE = 32
HIDDEN_DIM = 8
OUTPUT_SIZE = 5


class TestNetwork:
    @pytest.mark.parametrize("in_c", [3, HIDDEN_DIM])
    def test_row_lstm_forward_shape(self, in_c: int):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = RowLSTM(in_c, HIDDEN_DIM, 3)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))

    @pytest.mark.parametrize("in_c", [3, HIDDEN_DIM])
    def test_diagonal_lstm_cell_forward_shape(self, in_c: int):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)
        h = torch.zeros(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, 1)
        c = torch.zeros(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, 1)

        model = DiagonalLSTMCell(in_c, HIDDEN_DIM, 2)

        y = model(x, h, c)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))

    @pytest.mark.parametrize("in_c", [3, HIDDEN_DIM])
    @pytest.mark.parametrize("bidirectional", [False, True])
    def test_diagonal_lstm_forward_shape(self, in_c: int, bidirectional: bool):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = DiagonalLSTM(
            in_c,
            HIDDEN_DIM,
            2,
            bidirectional=bidirectional,
        )

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))

    @pytest.mark.parametrize("type,kernel", [("row", 3), ("diag", 2)])
    @pytest.mark.parametrize("bidirectional", [False, True])
    def test_bottleneck_forward_shape(
        self,
        type: str,
        kernel: int,
        bidirectional: bool,
    ):
        x = torch.rand(BATCH_SIZE, 2 * HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)

        model = BottleNeck(
            HIDDEN_DIM,
            kernel=kernel,
            type=type,
            bidirectional=bidirectional,
        )

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, 2 * HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("in_c", [1, 3])
    @pytest.mark.parametrize("rnn_type", ["row", "diag"])
    @pytest.mark.parametrize("bidirectional", [False, True])
    def test_forward_shape(self, in_c: int, rnn_type: str, bidirectional: bool):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = PixelRNN(
            in_c,
            256,
            HIDDEN_DIM,
            num_layer=6,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
        )

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, in_c, 256, IMAGE_SIZE, IMAGE_SIZE))

    @pytest.mark.parametrize("in_c", [1, 3])
    @pytest.mark.parametrize("rnn_type", ["row", "diag"])
    @pytest.mark.parametrize("bidirectional", [False, True])
    def test_multiscale_forward_shape(
        self,
        in_c: int,
        rnn_type: str,
        bidirectional: bool,
    ):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        encoder = PixelRNN(
            in_c,
            256,
            HIDDEN_DIM,
            num_layer=6,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
        )
        decoder = PixelRNN(
            in_c,
            256,
            HIDDEN_DIM,
            num_layer=6,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
        )

        model = MultiScalePixelRNN(encoder, decoder)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, in_c, 256, IMAGE_SIZE, IMAGE_SIZE))
