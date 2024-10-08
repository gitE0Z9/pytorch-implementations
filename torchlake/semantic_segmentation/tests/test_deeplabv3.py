import pytest
import torch

from ..models.deeplabv3 import ASPP, DeepLabV3
from ..models.deeplabv3.network import CascadeASPP


class TestDeepLabV3:
    @pytest.mark.parametrize("is_train", [True, False])
    @pytest.mark.parametrize("neck_type", ["parallel", "cascade"])
    def test_forward_shape(self, is_train: bool, neck_type: str):
        x = torch.rand((2, 3, 321, 321))

        model = DeepLabV3(21, neck_type=neck_type)
        if is_train:
            model.train()
        else:
            model.eval()

        y = model(x)

        assert y.shape == torch.Size((2, 21, 321, 321))

    def test_backward(self):
        x = torch.rand((2, 3, 321, 321))
        y = torch.randint(0, 21, (2, 321, 321))

        model = DeepLabV3(21)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)


class TestASPP:
    def test_forward_shape(self):
        x = torch.rand((2, 32, 41, 41))

        model = ASPP(32, 32, 21, [6, 12])
        y = model(x)

        assert y.shape == torch.Size((2, 21, 41, 41))


class TestCascadeASPP:
    def test_forward_shape(self):
        x = torch.rand((1, 2048, 41, 41))

        model = CascadeASPP()
        y = model(x)

        assert y.shape == torch.Size((1, 256, 41, 41))
