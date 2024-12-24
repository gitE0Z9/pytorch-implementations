import torch

from ..models.deit.loss import HardDistillation
from ..models.deit.model import DeiT


class TestModel:
    def test_forward_shape(self):
        x = torch.rand((2, 3, 224, 224))

        model = DeiT(output_size=21)

        cls, dist = model(x)

        assert cls.shape == torch.Size((2, 21))
        assert dist.shape == torch.Size((2, 21))


class TestLoss:
    def test_forward(self):
        x = torch.rand(2, 10, requires_grad=True)
        tx = torch.rand(2, 10)
        y = torch.randint(10, (2,))

        criterion = HardDistillation()
        loss = criterion(x, tx, y)

        assert not torch.isnan(loss)

    def test_backward(self):
        x = torch.rand(2, 10, requires_grad=True)
        tx = torch.rand(2, 10)
        y = torch.randint(10, (2,))

        criterion = HardDistillation()
        loss = criterion(x, tx, y)

        loss.backward()
