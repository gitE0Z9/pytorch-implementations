import torch

from ..models.hinton.loss import KLDLoss


class TestKLDLoss:
    def test_forward(self):
        x = torch.rand(2, 10, requires_grad=True)
        tx = torch.rand(2, 10)
        y = torch.randint(10, (2,))

        criterion = KLDLoss()
        loss = criterion(x, tx, y)

        assert not torch.isnan(loss)

    def test_backward(self):
        x = torch.rand(2, 10, requires_grad=True)
        tx = torch.rand(2, 10)
        y = torch.randint(10, (2,))

        criterion = KLDLoss()
        loss = criterion(x, tx, y)

        loss.backward()
