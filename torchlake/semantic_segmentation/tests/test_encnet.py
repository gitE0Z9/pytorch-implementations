import pytest
import torch

from torchlake.semantic_segmentation.models.deeplabv2.network import (
    deeplab_v2_style_resnet,
)

from ..models.encnet.model import EncNet
from ..models.encnet.network import EncodingModule2d
from ..models.encnet.loss import EncNetLoss

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 224
FEATURE_MAP_SIZE = IMAGE_SIZE // 32
NUM_CLASS = 21


class TestNetwork:
    @pytest.mark.parametrize("output_latent", (True, False))
    def test_encoding_module_2d_forward_shape(self, output_latent: bool):
        K = 4
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = EncodingModule2d(HIDDEN_DIM, k=K)

        y = model(x, output_latent=output_latent)
        if output_latent:
            y, z = y
            assert z.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM))

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize(
        "is_training,target_layers", ((True, ("3_1", "4_1")), (False, ("4_1",)))
    )
    def test_encnet_forward_shape(self, is_training: bool, target_layers):
        K = 32
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        backbone.fix_target_layers(target_layers)
        model = EncNet(backbone, NUM_CLASS, k=K)

        if is_training:
            model.train()
        else:
            model.eval()

        y = model(x)
        if is_training:
            y, z, aux = y
            assert aux.shape == torch.Size(
                (BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE)
            )
            assert z.shape == torch.Size((BATCH_SIZE, NUM_CLASS))

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))


class TestLoss:
    def test_encnet_loss_forward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        backbone.fix_target_layers(("3_1", "4_1"))
        model = EncNet(backbone, NUM_CLASS, k=32)
        model.train()

        yhat = model(x)
        criterion = EncNetLoss()
        loss = criterion(*yhat, y)

        assert not torch.isnan(loss)

    def test_encnet_loss_backward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        backbone.fix_target_layers(("3_1", "4_1"))
        model = EncNet(backbone, NUM_CLASS, k=32)
        model.train()

        yhat = model(x)
        criterion = EncNetLoss()
        loss = criterion(*yhat, y)

        loss.backward()
