from typing import Sequence

import pytest
import torch

from ..models.pixelcnn_pp.loss import DiscretizedLogisticMixture
from ..models.pixelcnn_pp.model import PixelCNNPP
from ..models.pixelcnn_pp.network import (
    DownsampleLayer,
    DownwardAndRightwardConv2d,
    DownwardAndRightwardConvTranspose2d,
    DownwardConv2d,
    DownwardConvTranspose2d,
    ResidualLayer,
    UpsampleLayer,
)

BATCH_SIZE = 2
IMAGE_SIZE = 32
NUM_MIXTURE = 5
HIDDEN_DIM = 8
OUTPUT_SIZE = 5


class TestNetwork:
    @pytest.mark.parametrize("in_c", [3, HIDDEN_DIM])
    @pytest.mark.parametrize("kernel,padding", [((2, 3), (0, 1)), ((1, 3), (0, 1))])
    def test_downward_conv_forward_shape(
        self,
        in_c: int,
        kernel: Sequence[int],
        padding: Sequence[int],
    ):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = DownwardConv2d(in_c, HIDDEN_DIM, kernel, padding=padding)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))

    @pytest.mark.parametrize("in_c", [3, HIDDEN_DIM])
    @pytest.mark.parametrize("kernel,padding", [((2, 2), 0), ((2, 1), 0)])
    def test_downward_and_rightward_conv_forward_shape(
        self,
        in_c: int,
        kernel: Sequence[int],
        padding: Sequence[int],
    ):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = DownwardAndRightwardConv2d(in_c, HIDDEN_DIM, kernel, padding=padding)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))

    @pytest.mark.parametrize("in_c", [3, HIDDEN_DIM])
    @pytest.mark.parametrize("kernel", [(2, 3)])
    def test_downward_deconv_forward_shape(self, in_c: int, kernel: Sequence[int]):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = DownwardConvTranspose2d(in_c, HIDDEN_DIM, kernel)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))

    @pytest.mark.parametrize("in_c", [3, HIDDEN_DIM])
    @pytest.mark.parametrize("kernel", [(2, 2)])
    def test_downward_and_rightward_deconv_forward_shape(
        self,
        in_c: int,
        kernel: Sequence[int],
    ):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = DownwardAndRightwardConvTranspose2d(in_c, HIDDEN_DIM, kernel)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))

    @pytest.mark.parametrize("in_c", [3, HIDDEN_DIM])
    def test_downsample_layer_forward_shape(self, in_c: int):
        v = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)
        h = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = DownsampleLayer(in_c, HIDDEN_DIM)

        v, h = model(v, h)

        assert v.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE // 2, IMAGE_SIZE // 2)
        )
        assert h.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE // 2, IMAGE_SIZE // 2)
        )

    @pytest.mark.parametrize("in_c", [3, HIDDEN_DIM])
    def test_upsample_layer_forward_shape(self, in_c: int):
        v = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)
        h = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)

        model = UpsampleLayer(in_c, HIDDEN_DIM)

        v, h = model(v, h)

        assert v.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE * 2, IMAGE_SIZE * 2)
        )
        assert h.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE * 2, IMAGE_SIZE * 2)
        )

    @pytest.mark.parametrize("is_upside", [True, False])
    def test_residual_layer_forward_shape(self, is_upside: bool):
        v = torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        h = torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
        av = (
            torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
            if is_upside
            else None
        )
        ah = (
            torch.rand(BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE)
            if is_upside
            else None
        )

        model = ResidualLayer(HIDDEN_DIM, HIDDEN_DIM, is_upside=is_upside)

        v, h = model(v, h, av, ah)

        assert v.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))
        assert h.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE, IMAGE_SIZE))


class TestLoss:
    @pytest.mark.parametrize("in_c", [1, 3])
    def test_forward(self, in_c: int):
        num_component = 1 + 2 * in_c + sum(range(in_c))
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)
        m = torch.nn.Conv2d(in_c, NUM_MIXTURE * num_component, 1)
        m.requires_grad_()
        yhat = m(x).unflatten(1, (NUM_MIXTURE, num_component))

        criterion = DiscretizedLogisticMixture(in_c)
        loss = criterion(yhat, x)

        assert not torch.isnan(loss)

    # @pytest.mark.parametrize("in_c", [1, 3])
    # def test_backward(self, in_c: int):
    #     num_component = 1 + 2 * in_c + sum(range(in_c))
    #     x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)
    #     m = torch.nn.Conv2d(in_c, NUM_MIXTURE * num_component, 1)
    #     m.requires_grad_()
    #     yhat = m(x).unflatten(1, (NUM_MIXTURE, num_component))

    #     criterion = DiscretizedLogisticMixture(in_c)
    #     loss = criterion(yhat, x)
    #     loss.backward()


class TestModel:
    @pytest.mark.parametrize("in_c", [1, 3])
    @pytest.mark.parametrize(
        "conditional,conditional_shape",
        (
            (None, None),
            (torch.rand((BATCH_SIZE, 10)), (10,)),
            # since image size of conditional could be different
            # the alignment solution could depend
            # (
            #     torch.rand((BATCH_SIZE, 10, IMAGE_SIZE, IMAGE_SIZE)),
            #     (10, IMAGE_SIZE, IMAGE_SIZE),
            # ),
        ),
    )
    def test_forward_shape(
        self,
        in_c: int,
        conditional: torch.Tensor | None,
        conditional_shape: Sequence[int] | None,
    ):
        x = torch.rand(BATCH_SIZE, in_c, IMAGE_SIZE, IMAGE_SIZE)
        num_component = 1 + 2 * in_c + sum(range(in_c))

        model = PixelCNNPP(
            in_c,
            HIDDEN_DIM,
            num_mixture=NUM_MIXTURE,
            num_block=5,
            conditional_shape=conditional_shape,
        )

        y = model(x, conditional)

        assert y.shape == torch.Size(
            (BATCH_SIZE, NUM_MIXTURE, num_component, IMAGE_SIZE, IMAGE_SIZE)
        )
