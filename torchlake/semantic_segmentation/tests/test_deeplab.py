import pytest
import torch

from ..models.deeplab.model import DeepLab
from ..models.deeplab.network import deeplab_style_vgg

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 321
DOWNSCALE_IMAGE_SIZE = IMAGE_SIZE // 8 + 1
NUM_CLASS = 21


class TestNetwork:
    @pytest.mark.parametrize("large_fov", (True, False))
    def test_deeplab_style_vgg_forward_shape(self, large_fov: bool):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        m = deeplab_style_vgg(
            "vgg16",
            trainable=False,
            large_fov=large_fov,
        )
        m.fix_target_layers(("1_1", "2_1", "3_1", "4_1", "6_1"))

        features = m(x)

        feature_dims = (
            m.hidden_dim_2x,
            m.hidden_dim_4x,
            m.hidden_dim_8x,
            m.hidden_dim_16x,
            m.feature_dim,
        )
        scales = (8, 16, 32)
        for f, d, _ in zip(features, feature_dims, scales):
            assert f.shape[:2] == torch.Size((BATCH_SIZE, d))


class TestModel:
    @pytest.mark.parametrize(
        "is_train,expected",
        [
            [True, DOWNSCALE_IMAGE_SIZE],
            [False, IMAGE_SIZE],
        ],
    )
    def test_deeplab_forward_shape(self, is_train: bool, expected: int):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_style_vgg("vgg16", trainable=False)
        model = DeepLab(backbone, NUM_CLASS)
        if is_train:
            model.train()
        else:
            model.eval()

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, expected, expected))

    def test_backward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        y = torch.randint(
            0, NUM_CLASS, (BATCH_SIZE, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

        backbone = deeplab_style_vgg("vgg16", trainable=True)
        model = DeepLab(backbone, NUM_CLASS)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)
