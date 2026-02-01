from typing import Callable
import pytest
import torch
from torch import nn

from ..models.deeplabv2.model import DeepLabV2
from ..models.deeplabv2.network import (
    ASPP,
    ShallowASPP,
    deeplab_v2_style_vgg,
    deeplab_v2_style_resnet,
)

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 321
DOWNSCALE_IMAGE_SIZE = IMAGE_SIZE // 8 + 1
NUM_CLASS = 21


class TestNetwork:
    def test_deeplab_v2_style_vgg_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = deeplab_v2_style_vgg("vgg16", trainable=False)
        features = model(x, ("1_1", "2_1", "3_1", "4_1", "6_1"))

        dims = (
            model.hidden_dim_2x,
            model.hidden_dim_4x,
            model.hidden_dim_8x,
            model.hidden_dim_16x,
            model.feature_dim,
        )
        for f, d in zip(features, dims):
            assert f.shape[:2] == torch.Size((BATCH_SIZE, d))

    @pytest.mark.parametrize("output_stride", (8, 16))
    def test_deeplab_v2_style_resnet_forward_shape(self, output_stride: int):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = deeplab_v2_style_resnet(
            "resnet50",
            trainable=False,
            output_stride=output_stride,
        )
        features = model(x, ("0_1", "1_1", "2_1", "3_1", "4_1"))

        dims = (
            model.hidden_dim_stem,
            model.hidden_dim_4x,
            model.hidden_dim_8x,
            model.hidden_dim_16x,
            model.hidden_dim_32x,
        )
        for f, d in zip(features, dims):
            assert f.shape[:2] == torch.Size((BATCH_SIZE, d))

    def test_aspp_forward_shape(self):
        x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

        model = ASPP(HIDDEN_DIM, HIDDEN_DIM, NUM_CLASS, [6, 12])
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, NUM_CLASS, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

    def test_shallow_aspp_forward_shape(self):
        x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

        model = ShallowASPP(HIDDEN_DIM, NUM_CLASS, [6, 12])
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, NUM_CLASS, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize(
        "is_train,expected",
        [
            [True, DOWNSCALE_IMAGE_SIZE],
            [False, IMAGE_SIZE],
        ],
    )
    @pytest.mark.parametrize(
        "fe_constructor",
        [
            lambda: deeplab_v2_style_vgg("vgg16", trainable=False),
            lambda: deeplab_v2_style_resnet("resnet101", trainable=False),
        ],
    )
    def test_deeplab_v2_forward_shape(
        self,
        is_train: bool,
        expected: int,
        fe_constructor: Callable[[None], nn.Module],
    ):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        fe = fe_constructor()
        model = DeepLabV2(fe, output_size=NUM_CLASS)
        if is_train:
            model.train()
        else:
            model.eval()

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, expected, expected))

    @pytest.mark.parametrize(
        "fe",
        [
            deeplab_v2_style_vgg("vgg16", trainable=False),
            deeplab_v2_style_resnet("resnet101", trainable=False),
        ],
    )
    def test_deeplab_v2_backward(self, fe: nn.Module):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        y = torch.randint(
            0, NUM_CLASS, (BATCH_SIZE, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

        model = DeepLabV2(fe, output_size=NUM_CLASS)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)
