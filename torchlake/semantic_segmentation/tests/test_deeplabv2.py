import pytest
import torch

from ..models.deeplabv2 import DeepLabV2, ASPP
from ...common.models import VGGFeatureExtractor


class TestDeepLabV2:
    @pytest.mark.parametrize(
        "is_train,expected",
        [
            [True, 321 // 8 + 1],
            [False, 321],
        ],
    )
    def test_forward_shape(self, is_train: bool, expected: int):
        x = torch.rand((1, 3, 321, 321))

        fe = VGGFeatureExtractor("vgg16", "maxpool", trainable=False)
        model = DeepLabV2(fe, output_size=21)
        if is_train:
            model.train()
        else:
            model.eval()

        y = model(x)

        assert y.shape == torch.Size((1, 21, expected, expected))

    def test_backward(self):
        x = torch.rand((1, 3, 321, 321))
        y = torch.randint(0, 21, (1, 41, 41))

        fe = VGGFeatureExtractor("vgg16", "maxpool", trainable=False)
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
