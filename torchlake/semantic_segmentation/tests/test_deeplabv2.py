import pytest
import torch
from torch import nn

from ...common.models import ResNetFeatureExtractor, VGGFeatureExtractor
from ..models.deeplabv2 import ASPP, DeepLabV2
from ..models.deeplabv2.network import ShallowASPP


class TestDeepLabV2:
    @pytest.mark.parametrize(
        "is_train,expected",
        [
            [True, 321 // 8 + 1],
            [False, 321],
        ],
    )
    @pytest.mark.parametrize(
        "fe",
        [
            VGGFeatureExtractor("vgg16", "maxpool", trainable=False),
            ResNetFeatureExtractor("resnet101", "block", trainable=False),
        ],
    )
    def test_forward_shape(self, is_train: bool, expected: int, fe: nn.Module):
        x = torch.rand((1, 3, 321, 321))

        model = DeepLabV2(fe, output_size=21)
        if is_train:
            model.train()
        else:
            model.eval()

        y = model(x)

        assert y.shape == torch.Size((1, 21, expected, expected))

    @pytest.mark.parametrize(
        "fe",
        [
            VGGFeatureExtractor("vgg16", "maxpool", trainable=False),
            ResNetFeatureExtractor("resnet101", "block", trainable=False),
        ],
    )
    def test_backward(self, fe: nn.Module):
        x = torch.rand((1, 3, 321, 321))
        y = torch.randint(0, 21, (1, 41, 41))

        model = DeepLabV2(fe, output_size=21)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)


class TestASPP:
    def test_forward_shape(self):
        x = torch.rand((1, 32, 41, 41))

        model = ASPP(32, 32, 21, [6, 12])
        y = model(x)

        assert y.shape == torch.Size((1, 21, 41, 41))


class TestShallowASPP:
    def test_forward_shape(self):
        x = torch.rand((1, 32, 41, 41))

        model = ShallowASPP(32, 21, [6, 12])
        y = model(x)

        assert y.shape == torch.Size((1, 21, 41, 41))
