import pytest
import torch

from ..models.dense_aspp.model import DeepLabV3WithDenseASPP
from ..models.dense_aspp.network import DenseASPP
from ..models.deeplabv2.network import deeplab_v2_style_resnet

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 320
DOWNSCALE_IMAGE_SIZE = IMAGE_SIZE // 8
NUM_CLASS = 21


class TestNetwork:
    def test_dense_aspp_forward_shape(self):
        in_c = HIDDEN_DIM + 1
        x: torch.Tensor = torch.rand(
            (BATCH_SIZE, in_c, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

        model = DenseASPP(in_c, HIDDEN_DIM, (3, 6, 12, 18, 24))
        y = model(x)

        assert y.shape == torch.Size(
            (
                BATCH_SIZE,
                in_c + HIDDEN_DIM * len(model.dilations),
                DOWNSCALE_IMAGE_SIZE,
                DOWNSCALE_IMAGE_SIZE,
            )
        )


class TestModel:
    def test_deeplab_v3_with_dense_aspp_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        backbone.fix_target_layers(("4_1",))
        model = DeepLabV3WithDenseASPP(backbone, output_size=NUM_CLASS)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
