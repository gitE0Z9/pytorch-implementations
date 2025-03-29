import pytest
import torch

from ..models.pointnet.loss import PointNetLoss
from ..models.pointnet.model import PointNet
from ..models.pointnet.network import TNet, TransformModule

BATCH_SIZE = 2
NUM_POINT = 32
HIDDEN_DIM = 8
OUTPUT_SIZE = 5


class TestNetwork:
    def test_forward_shape_tnet(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, NUM_POINT))

        model = TNet(HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, HIDDEN_DIM))

    def test_forward_shape_transform_module(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, NUM_POINT))

        model = TransformModule(HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, HIDDEN_DIM, NUM_POINT))


class TestModel:
    @pytest.mark.parametrize("is_training", [True, False])
    def test_forward_shape(self, is_training: bool):
        x = torch.rand((BATCH_SIZE, NUM_POINT, 3))

        model = PointNet(output_size=OUTPUT_SIZE)
        if is_training:
            model.train()
        else:
            model.eval()

        y = model(x)

        if is_training:
            y, (t1, t2) = y
            assert t1.shape == torch.Size((BATCH_SIZE, 3, 3))
            assert t2.shape == torch.Size((BATCH_SIZE, 64, 64))

        assert y.shape == torch.Size((BATCH_SIZE, OUTPUT_SIZE))


class TestLoss:
    def test_forward(self):
        yhat = torch.rand((BATCH_SIZE, OUTPUT_SIZE))
        y = torch.randint(0, OUTPUT_SIZE, (BATCH_SIZE,))
        t = (
            torch.rand((BATCH_SIZE, 3, 3)),
            torch.rand((BATCH_SIZE, HIDDEN_DIM, HIDDEN_DIM)),
        )
        criterion = PointNetLoss()

        loss = criterion(yhat, t, y)

        assert not torch.isnan(loss)

    def test_backward(self):
        yhat = torch.rand((BATCH_SIZE, OUTPUT_SIZE)).requires_grad_()
        y = torch.randint(0, OUTPUT_SIZE, (BATCH_SIZE,))
        t = (
            torch.rand((BATCH_SIZE, 3, 3)).requires_grad_(),
            torch.rand((BATCH_SIZE, HIDDEN_DIM, HIDDEN_DIM)).requires_grad_(),
        )
        criterion = PointNetLoss()

        loss = criterion(yhat, t, y)
        loss.backward()
