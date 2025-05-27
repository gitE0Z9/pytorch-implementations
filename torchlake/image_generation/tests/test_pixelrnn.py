import pytest
import torch

from ..models.pixelrnn.model import PixelRNN
from ..models.pixelrnn.network import BottleNeck, RowLSTM, DiagonalLSTM, DiagonalBiLSTM

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
    def test_diagonal_lstm_forward_shape(self, in_c: int):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = DiagonalLSTM(in_c, HIDDEN_DIM, 2)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))

    @pytest.mark.parametrize("in_c", [3, HIDDEN_DIM])
    def test_diagonal_bilstm_forward_shape(self, in_c: int):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = DiagonalBiLSTM(in_c, HIDDEN_DIM, 2)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))

    @pytest.mark.parametrize("type,kernel", [("row", 3), ("diag", 2)])
    def test_bottleneck_forward_shape(self, type: str, kernel: int):
        x = torch.rand(BATCH_SIZE, 2 * HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)

        model = BottleNeck(HIDDEN_DIM, kernel=kernel, type=type)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, 2 * HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("in_c", [1, 3])
    @pytest.mark.parametrize("rnn_type", ["row", "diag"])
    def test_forward_shape(self, in_c: int, rnn_type: str):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = PixelRNN(in_c, 256, HIDDEN_DIM, num_layer=6, rnn_type=rnn_type)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, in_c, 256, IMAGE_SIZE, IMAGE_SIZE))
