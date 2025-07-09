from typing import Sequence
import pytest
import torch

from ..models.gated_pixelcnn.model import GatedPixelCNN
from ..models.gated_pixelcnn.network import DownwardConv2d, GatedLayer

BATCH_SIZE = 2
IMAGE_SIZE = 32
HIDDEN_DIM = 8
OUTPUT_SIZE = 5


class TestNetwork:
    # kernel 5 is not supported, for 2 is hard to pad
    @pytest.mark.parametrize("kernel", (3, 7))
    def test_downward_conv_2d_forward_shape(self, kernel: int):
        x = torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

        model = DownwardConv2d(3, HIDDEN_DIM, kernel)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))

    @pytest.mark.parametrize(
        "conditional,conditional_shape",
        (
            (None, None),
            (torch.rand((BATCH_SIZE, 10, 1, 1)), (10,)),
            (
                torch.rand((BATCH_SIZE, 10, IMAGE_SIZE, IMAGE_SIZE)),
                (10, IMAGE_SIZE, IMAGE_SIZE),
            ),
        ),
    )
    def test_gated_layer_forward_shape(
        self,
        conditional: torch.Tensor | None,
        conditional_shape: Sequence[int] | None,
    ):
        v = torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        h = torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)

        model = GatedLayer(HIDDEN_DIM, 3, conditional_shape=conditional_shape)

        yv, yh = model(v, h, conditional)

        assert yv.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))
        assert yh.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))


class TestModel:
    @pytest.mark.parametrize("in_c", [1, 3])
    @pytest.mark.parametrize(
        "conditional,conditional_shape",
        (
            (None, None),
            (torch.rand((BATCH_SIZE, 10)), (10,)),
            (
                torch.rand((BATCH_SIZE, 10, IMAGE_SIZE, IMAGE_SIZE)),
                (10, IMAGE_SIZE, IMAGE_SIZE),
            ),
        ),
    )
    def test_forward_shape(
        self,
        in_c: int,
        conditional: torch.Tensor | None,
        conditional_shape: Sequence[int] | None,
    ):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = GatedPixelCNN(
            in_c,
            256,
            HIDDEN_DIM,
            num_layers=6,
            conditional_shape=conditional_shape,
        )

        y = model(x, conditional)

        assert y.shape == torch.Size((BATCH_SIZE, in_c, 256, IMAGE_SIZE, IMAGE_SIZE))
