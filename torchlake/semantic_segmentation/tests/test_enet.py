import pytest
import torch
from torch import nn


from ..models.enet.model import ENet
from ..models.enet.network import Stem, BottleNeck, UpsamplingBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 512
FEATURE_MAP_SIZE = 16
NUM_CLASS = 21


class TestNetwork:
    def test_stem_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = Stem(INPUT_CHANNEL, HIDDEN_DIM)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, INPUT_CHANNEL + HIDDEN_DIM, IMAGE_SIZE // 2, IMAGE_SIZE // 2)
        )

    @pytest.mark.parametrize(
        "name,stride,dilation,asymmetric",
        (
            ("conv", 1, 1, False),
            ("strided-conv", 2, 1, False),
            ("dilated-conv", 1, 2, False),
            ("asymmetric-conv", 1, 1, True),
        ),
    )
    def test_bottleneck_forward_shape(self, name, stride, dilation, asymmetric):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = BottleNeck(
            HIDDEN_DIM,
            HIDDEN_DIM,
            stride=stride,
            dilation=dilation,
            asymmetric=asymmetric,
        )
        y = model(x)
        s = FEATURE_MAP_SIZE // stride
        if stride > 1:
            y, indices = y
            assert indices.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, s, s))

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, s, s))

    def test_upsampling_block_forward_shape(self):
        stride, in_c, out_c = 2, HIDDEN_DIM, HIDDEN_DIM // 2
        x = torch.rand((BATCH_SIZE, in_c, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))
        pooling_indices = torch.randint(
            0,
            FEATURE_MAP_SIZE**2,
            (BATCH_SIZE, out_c, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE),
        )

        model = UpsamplingBlock(in_c, out_c)
        y = model(x, pooling_indices)

        s = FEATURE_MAP_SIZE * stride
        assert y.shape == torch.Size((BATCH_SIZE, out_c, s, s))


class TestModel:
    def test_enet_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = ENet(INPUT_CHANNEL, NUM_CLASS)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))

    def test_enet_backward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        model = ENet(INPUT_CHANNEL, NUM_CLASS)
        criterion = nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)
