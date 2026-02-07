import torch
from torch import nn

from ..models.deeplabv2 import deeplab_v2_style_resnet

from ..models.pan.model import PAN
from ..models.pan.network import FPA, GAU

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 512
FEATURE_MAP_SIZE = 16
NUM_CLASS = 21


class TestNetwork:
    def test_fpa_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = FPA(
            HIDDEN_DIM,
            HIDDEN_DIM,
            HIDDEN_DIM,
            (
                (1,) * 3,
                (1,) * 3,
            ),
        )
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )

    def test_gau_forward_shape(self):
        shallow_x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )
        deep_x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE // 2, FEATURE_MAP_SIZE // 2)
        )

        model = GAU(HIDDEN_DIM, HIDDEN_DIM + 2)
        y = model(shallow_x, deep_x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    def test_pan_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        model = PAN(backbone, NUM_CLASS, hidden_dim=HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))

    def test_pan_backward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        model = PAN(backbone, NUM_CLASS, hidden_dim=HIDDEN_DIM)
        criterion = nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)
