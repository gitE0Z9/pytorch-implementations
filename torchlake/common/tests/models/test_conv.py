import torch
import pytest

from ...models.conv import ConvBNReLU, ConvINReLU, CausalConv1d


class TestConvBNReLU:
    @pytest.mark.parametrize(
        "input_shape,output_shape,dimension",
        [
            [(8, 16, 7), (8, 32, 7), "1d"],
            [(8, 16, 7, 7), (8, 32, 7, 7), "2d"],
            [(8, 16, 7, 7, 7), (8, 32, 7, 7, 7), "3d"],
        ],
    )
    def test_output_shape(
        self,
        input_shape: tuple[int],
        output_shape: tuple[int],
        dimension: str,
    ):
        x = torch.randn(*input_shape)

        model = ConvBNReLU(16, 32, 3, padding=1, dimension=dimension)

        y = model(x)

        assert y.shape == torch.Size(output_shape)


class TestConvINReLU:
    @pytest.mark.parametrize(
        "input_shape,output_shape,dimension",
        [
            [(8, 16, 7), (8, 32, 7), "1d"],
            [(8, 16, 7, 7), (8, 32, 7, 7), "2d"],
            [(8, 16, 7, 7, 7), (8, 32, 7, 7, 7), "3d"],
        ],
    )
    def test_output_shape(
        self,
        input_shape: tuple[int],
        output_shape: tuple[int],
        dimension: str,
    ):
        x = torch.randn(*input_shape)

        model = ConvINReLU(16, 32, 3, padding=1, dimension=dimension)

        y = model(x)

        assert y.shape == torch.Size(output_shape)


BATCH_SIZE = 2
HIDDEN_DIM = 8
SEQ_LEN = 1024


class TestCausalConv:
    def test_causal_conv_1d_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, SEQ_LEN))
        model = CausalConv1d(HIDDEN_DIM, HIDDEN_DIM + 2, 2, padding=1)
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM + 2, SEQ_LEN))
