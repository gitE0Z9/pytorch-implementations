import pytest
import torch

from ..models.r_aspp import MobileNetV2Seg


class TestMobileNetV2Seg:
    @pytest.mark.parametrize("is_train", [True, False])
    def test_forward_shape(self, is_train: bool):
        x = torch.rand((2, 3, 321, 321))

        model = MobileNetV2Seg(21)
        if is_train:
            model.train()
        else:
            model.eval()

        y = model(x)

        assert y.shape == torch.Size((2, 21, 321, 321))

    def test_backward(self):
        x = torch.rand((2, 3, 321, 321))
        y = torch.randint(0, 21, (2, 321, 321))

        model = MobileNetV2Seg(21)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)
