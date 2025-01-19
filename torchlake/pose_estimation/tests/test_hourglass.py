import pytest
import torch

from ..models.hourglass.model import StackedHourglass
from ..models.hourglass.network import AuxiliaryHead, Hourglass2d
from ..models.hourglass.loss import StackedHourglassLoss


class TestStackedHourglass:
    @pytest.mark.parametrize("num_stack", [1, 2, 4, 8])
    @pytest.mark.parametrize("num_resblock", [1, 2])
    def test_forward_shape(self, num_stack, num_resblock):
        x = torch.rand((2, 3, 224, 224))

        model = StackedHourglass(
            output_size=16,
            num_stack=num_stack,
            num_resblock=num_resblock,
        )

        y = model(x)

        assert y.shape == torch.Size((2, num_stack, 16, 56, 56))


class TestHourglass2d:
    def test_forward_shape(self):
        x = torch.rand((2, 8, 56, 56))

        model = Hourglass2d(8, 4)

        y = model(x)

        assert y.shape == torch.Size((2, 8, 56, 56))


class TestAuxiliaryHead:
    @pytest.mark.parametrize("output_neck", [True, False])
    def test_forward_shape(self, output_neck):
        x = torch.rand((2, 8, 7, 7))

        model = AuxiliaryHead(8, 4)

        y = model(x, output_neck=output_neck)

        if output_neck:
            assert y[0].shape == torch.Size((2, 4, 7, 7))
            assert y[1].shape == torch.Size((2, 8, 7, 7))
        else:
            assert y.shape == torch.Size((2, 4, 7, 7))


class TestLoss:
    def test_forward(self):
        # batch, stack, num point, h, w
        x = torch.rand(2, 3, 3, 7, 7)
        # batch, num point, 2
        y = torch.LongTensor(
            [
                [
                    [1, 1],
                    [2, 2],
                    [3, 3],
                ],
                [
                    [1, 2],
                    [2, 4],
                    [3, 6],
                ],
            ]
        )

        criterion = StackedHourglassLoss()
        loss = criterion(x, y)

        assert not torch.isnan(loss)
