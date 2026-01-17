import pytest
import torch

from ..models.deeplabv3p.model import DeepLabV3Plus
from ..models.deeplabv2.network import deeplab_v2_style_resnet
from ..models.deeplabv3p.network import Decoder

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 321
DOWNSCALE_IMAGE_SIZE = IMAGE_SIZE // 8 + 1
NUM_CLASS = 21


class TestNetwork:
    def test_decoder_forward_shape(self):
        z_shallow = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE // 4, IMAGE_SIZE // 4)
        )
        z_deep = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, IMAGE_SIZE // 16, IMAGE_SIZE // 16)
        )

        model = Decoder(HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, NUM_CLASS)
        y = model(z_shallow, z_deep)

        assert y.shape == torch.Size(
            (BATCH_SIZE, NUM_CLASS, IMAGE_SIZE // 4, IMAGE_SIZE // 4)
        )


class TestModel:
    @pytest.mark.parametrize("is_train", [True, False])
    def test_deeplab_v3_plus_forward_shape(self, is_train: bool):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        model = DeepLabV3Plus(backbone, output_size=NUM_CLASS)
        if is_train:
            model.train()
        else:
            model.eval()

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))

    def test_deeplab_v3_plus_backward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        model = DeepLabV3Plus(backbone, output_size=NUM_CLASS)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)
