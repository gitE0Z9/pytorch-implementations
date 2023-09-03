import torch
from yolov2.network import darknet19
from yolov2.detector import yolov2, yolov2_resnet
from torch.testing import assert_equal


class Test_yolov2():

    def test_output_shape(self):
        backbone = darknet19()
        model = yolov2(backbone, 5, 20)
        testx = torch.rand(2, 3, 416, 416)

        output = model(testx)
        assert_equal(output.shape, torch.Size([2, 125, 13, 13]))


class Test_yolov2_resnet18():

    def test_output_shape(self):
        model = yolov2_resnet(18, 5, 20)
        testx = torch.rand(2, 3, 416, 416)

        output = model(testx)
        assert_equal(output.shape, torch.Size([2, 125, 13, 13]))


class Test_yolov2_resnet34():

    def test_output_shape(self):
        model = yolov2_resnet(34, 5, 20)
        testx = torch.rand(2, 3, 416, 416)

        output = model(testx)
        assert_equal(output.shape, torch.Size([2, 125, 13, 13]))

