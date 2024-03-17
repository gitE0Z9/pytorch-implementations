import torch
from torch.testing import assert_close

from ..models.yolov1.detector import Yolov1, Yolov1Resnet


class TestYolov1:
    def test_output_shape(self):
        model = Yolov1(2, 30)
        test_x = torch.rand(2, 3, 448, 448)

        output: torch.Tensor = model(test_x)
        assert_close(output.shape, torch.Size([2, 30 + 2 * 5, 7, 7]))


class TestYolov1Resnet:
    def test_output_shape(self):
        model = Yolov1Resnet(18, 2, 30)
        test_x = torch.rand(2, 3, 448, 448)

        output: torch.Tensor = model(test_x)
        assert_close(output.shape, torch.Size([2, 30 + 2 * 5, 7, 7]))
