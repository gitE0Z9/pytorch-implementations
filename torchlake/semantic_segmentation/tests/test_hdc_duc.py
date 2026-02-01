import pytest
import torch

from ..models.hdc_duc.model import HDCDUC
from ..models.hdc_duc.network import DUC, tusimple_style_resnet

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 320
NUM_CLASS = 21


class TestNetwork:
    @pytest.mark.parametrize("output_stride", (8, 16))
    def test_tusimple_style_resnet_forward_shape(self, output_stride: int):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = tusimple_style_resnet(
            "resnet101",
            trainable=False,
            output_stride=output_stride,
        )
        model.fix_target_layers(("4_1",))
        y = model(x).pop()

        assert y.shape == torch.Size(
            (
                BATCH_SIZE,
                model.hidden_dim_32x,
                IMAGE_SIZE // output_stride,
                IMAGE_SIZE // output_stride,
            )
        )

    @pytest.mark.parametrize("output_stride", (8, 16))
    def test_duc_forward_shape(self, output_stride: int):
        x = torch.rand(
            (
                BATCH_SIZE,
                HIDDEN_DIM,
                IMAGE_SIZE // output_stride,
                IMAGE_SIZE // output_stride,
            )
        )

        model = DUC(
            HIDDEN_DIM,
            NUM_CLASS,
            output_stride,
            dilations=(6, 12, 18),
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))


class TestModel:
    @pytest.mark.parametrize("output_stride", (8, 16))
    def test_hdc_duc_forward_shape(self, output_stride: int):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = tusimple_style_resnet(
            "resnet101",
            trainable=False,
            output_stride=output_stride,
        )
        model = HDCDUC(
            backbone,
            NUM_CLASS,
            output_stride=output_stride,
        )
        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
