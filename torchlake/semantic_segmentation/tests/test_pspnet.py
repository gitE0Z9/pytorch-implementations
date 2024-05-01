import torch
from ..models import PspNet
from ..models.pspnet.network import PyramidPool2d
from ..models.pspnet.loss import PspLoss


class TestPspNet:
    def test_training_forward_shape(self):
        x = torch.rand((16, 3, 224, 224))

        model = PspNet(2048, 21)

        y, aux = model(x)

        assert y.shape == torch.Size((16, 21, 224, 224))
        assert aux.shape == torch.Size((16, 21, 224, 224))

    def test_eval_forward_shape(self):
        x = torch.rand((16, 3, 224, 224))

        model = PspNet(2048, 21).eval()

        y = model(x)

        assert y.shape == torch.Size((16, 21, 224, 224))


class TestPyramidPool2d:
    def test_forward_shape(self):
        x = torch.rand((16, 2048, 7, 7))

        model = PyramidPool2d(2048)

        y = model(x)

        assert y.shape == torch.Size((16, 2048 * 2, 7, 7))


class TestPspLoss:
    def test_forward(self):
        pred = torch.rand((16, 21, 224, 224))
        aux = torch.rand((16, 21, 224, 224))
        target = torch.randint(0, 21, (16, 224, 224))

        model = PspLoss()

        loss = model(pred, aux, target)

        assert not torch.isnan(loss)

    def test_backward(self):
        x = torch.rand((16, 3, 224, 224))
        target = torch.randint(0, 21, (16, 224, 224))
        model = PspNet(2048, 21).eval()
        model.train()

        y = model(x)
        model = PspLoss()

        loss = model(*y, target)
        loss.backward()
