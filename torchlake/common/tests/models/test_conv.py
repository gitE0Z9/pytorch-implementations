import torch
import pytest

from ...models import ConvBNReLU, ConvInReLU


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


class TestConvInReLU:
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

        model = ConvInReLU(16, 32, 3, padding=1, dimension=dimension)

        y = model(x)

        assert y.shape == torch.Size(output_shape)
