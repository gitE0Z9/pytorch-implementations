import pytest
import torch

from ..models.r_aspp.network import r_aspp_style_mobilenet
from ..models.lr_aspp.model import MobileNetV3Seg
from ..models.lr_aspp.network import LRASPP

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 512
FEATURE_MAP_SIZE_8X = IMAGE_SIZE // 8
FEATURE_MAP_SIZE_16X = IMAGE_SIZE // 16
NUM_CLASS = 21


class TestNetwork:
    def test_lr_aspp_forward_shape(self):
        shallow_x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE_8X, FEATURE_MAP_SIZE_8X)
        )
        deep_x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE_16X, FEATURE_MAP_SIZE_16X)
        )

        model = LRASPP(
            HIDDEN_DIM,
            HIDDEN_DIM,
            2,
            HIDDEN_DIM,
            NUM_CLASS,
            pool_kernel_size=(32, 32),
            pool_stride=(16, 16),
        )
        y = model(shallow_x, deep_x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, NUM_CLASS, FEATURE_MAP_SIZE_8X, FEATURE_MAP_SIZE_8X)
        )


class TestModel:
    @pytest.mark.parametrize("is_train", [True, False])
    def test_mobilenet_v3_seg_forward_shape(self, is_train: bool):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = r_aspp_style_mobilenet(
            "mobilenet_v3_large",
            trainable=False,
            dilation_size_16x=1,
            dilation_size_32x=2,
        )
        model = MobileNetV3Seg(
            backbone,
            NUM_CLASS,
            pool_kernel_size=(32, 32),
            pool_stride=(16, 16),
        )
        if is_train:
            model.train()
        else:
            model.eval()

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))

    def test_mobilenet_v3_seg_backward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        backbone = r_aspp_style_mobilenet(
            "mobilenet_v3_large",
            trainable=False,
            dilation_size_16x=1,
            dilation_size_32x=2,
        )
        model = MobileNetV3Seg(
            backbone,
            NUM_CLASS,
            pool_kernel_size=(32, 32),
            pool_stride=(16, 16),
        )
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)
