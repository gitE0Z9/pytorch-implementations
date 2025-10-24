import torch

from ..models.deit.loss import HardDistillation
from ..models.deit.model import DeiT

BATCH_SIZE = 2


class TestModel:
    def test_forward_shape(self):
        x = torch.rand((BATCH_SIZE, 3, 224, 224))

        model = DeiT(output_size=21)

        cls, dist = model(x)

        assert cls.shape == torch.Size((BATCH_SIZE, 21))
        assert dist.shape == torch.Size((BATCH_SIZE, 21))


class TestLoss:
    def test_forward(self):
        yhat = torch.rand(BATCH_SIZE, 10, requires_grad=True)
        yhat_for_t = torch.rand(BATCH_SIZE, 10, requires_grad=True)
        yhat_of_t = torch.rand(BATCH_SIZE, 10)
        y = torch.randint(10, (BATCH_SIZE,))

        criterion = HardDistillation()
        loss = criterion(yhat, yhat_for_t, yhat_of_t, y)

        assert not torch.isnan(loss)

    def test_backward(self):
        yhat = torch.rand(BATCH_SIZE, 10, requires_grad=True)
        yhat_for_t = torch.rand(BATCH_SIZE, 10, requires_grad=True)
        yhat_of_t = torch.rand(BATCH_SIZE, 10)
        y = torch.randint(10, (BATCH_SIZE,))

        criterion = HardDistillation()
        loss = criterion(yhat, yhat_for_t, yhat_of_t, y)

        loss.backward()
