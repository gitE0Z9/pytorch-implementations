import torch

from ..models.hinton.loss import KLDLoss

BATCH_SIZE = 2
OUTPUT_SIZE = 10


class TestLoss:
    def test_kld_loss_forward(self):
        x = torch.rand(BATCH_SIZE, OUTPUT_SIZE, requires_grad=True)
        tx = torch.rand(BATCH_SIZE, OUTPUT_SIZE)
        y = torch.randint(OUTPUT_SIZE, (BATCH_SIZE,))

        criterion = KLDLoss()
        loss = criterion(x, tx, y)

        assert not torch.isnan(loss)

    def test_kld_loss_backward(self):
        x = torch.rand(BATCH_SIZE, OUTPUT_SIZE, requires_grad=True)
        tx = torch.rand(BATCH_SIZE, OUTPUT_SIZE)
        y = torch.randint(OUTPUT_SIZE, (BATCH_SIZE,))

        criterion = KLDLoss()
        loss = criterion(x, tx, y)

        loss.backward()
