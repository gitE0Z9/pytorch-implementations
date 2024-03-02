import torch
from torch.testing import assert_close

from ..models.yolov2.detector import Yolov2, Yolov2Resnet
from ..models.yolov2.network import Darknet19


class TestYOLOv2:
    def test_output_shape(self):
        backbone = Darknet19()
        model = Yolov2(backbone, 5, 20)
        test_x = torch.rand(2, 3, 416, 416)

        output: torch.Tensor = model(test_x)
        assert_close(output.shape, torch.Size([2, 125, 13, 13]))


class TestYOLOv2ResNet18:
    def test_output_shape(self):
        model = Yolov2Resnet(18, 5, 20)
        test_x = torch.rand(2, 3, 416, 416)

        output: torch.Tensor = model(test_x)
        assert_close(output.shape, torch.Size([2, 125, 13, 13]))


class TestYOLOv2ResNet34:
    def test_output_shape(self):
        model = Yolov2Resnet(34, 5, 20)
        test_x = torch.rand(2, 3, 416, 416)

        output: torch.Tensor = model(test_x)
        assert_close(output.shape, torch.Size([2, 125, 13, 13]))
