import pytest
import torch

from torchlake.semantic_segmentation.models.deeplabv2.network import (
    deeplab_v2_style_resnet,
)

from ..models.pspnet import PSPLoss, PSPNet, PyramidPool2d

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 224
FEATURE_MAP_SIZE = IMAGE_SIZE // 32
NUM_CLASS = 21


class TestNetwork:
    def test_pyramid_pool_2d_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = PyramidPool2d(HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM * 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("is_training", ((True,), (False,)))
    def test_pspnet_forward_shape(self, is_training: bool):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        model = PSPNet(backbone, NUM_CLASS)

        if is_training:
            model.train()
            y, aux = model(x)
            assert y.shape == torch.Size(
                (BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE)
            )
            assert aux.shape == torch.Size(
                (BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE)
            )
        else:
            model.eval()
            y = model(x)
            assert y.shape == torch.Size(
                (BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE)
            )


class TestLoss:
    def test_psp_loss_forward(self):
        pred = torch.rand((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
        aux = torch.rand((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
        target = torch.randint(0, NUM_CLASS, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        criterion = PSPLoss()

        loss = criterion(pred, aux, target)

        assert not torch.isnan(loss)

    def test_psp_loss_backward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        target = torch.randint(0, NUM_CLASS, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        model = PSPNet(backbone, NUM_CLASS)
        model.train()

        y = model(x)

        criterion = PSPLoss()
        loss = criterion(*y, target)
        loss.backward()
