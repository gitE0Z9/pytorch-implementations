import torch

from ..models.pspnet import PSPNet, PSPLoss, PyramidPool2d


class TestPSPNet:
    def test_training_forward_shape(self):
        x = torch.rand((16, 3, 224, 224))

        model = PSPNet(2048, 21)
        model.train()

        y, aux = model(x)

        assert y.shape == torch.Size((16, 21, 224, 224))
        assert aux.shape == torch.Size((16, 21, 224, 224))

    def test_eval_forward_shape(self):
        x = torch.rand((16, 3, 224, 224))

        model = PSPNet(2048, 21).eval()

        y = model(x)

        assert y.shape == torch.Size((16, 21, 224, 224))


class TestPyramidPool2d:
    def test_forward_shape(self):
        x = torch.rand((16, 2048, 7, 7))

        model = PyramidPool2d(2048)

        y = model(x)

        assert y.shape == torch.Size((16, 2048 * 2, 7, 7))


class TestPSPLoss:
    def test_forward(self):
        pred = torch.rand((16, 21, 224, 224))
        aux = torch.rand((16, 21, 224, 224))
        target = torch.randint(0, 21, (16, 224, 224))

        model = PSPLoss()

        loss = model(pred, aux, target)

        assert not torch.isnan(loss)

    def test_backward(self):
        x = torch.rand((16, 3, 224, 224))
        target = torch.randint(0, 21, (16, 224, 224))
        model = PSPNet(2048, 21).eval()
        model.train()

        y = model(x)
        model = PSPLoss()

        loss = model(*y, target)
        loss.backward()
