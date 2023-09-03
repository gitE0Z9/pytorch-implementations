import torch
from yolov2.network import darknet19, Reorg, convblock, bottleneck
from torch.testing import assert_equal


class Test_darknet19():

    def test_output_shape(self):
        model = darknet19()
        testx = torch.rand(2, 3, 224, 224)

        output = model(testx)
        assert_equal(output.shape, torch.Size([2, 1000]))


class Test_Reorg():

    def test_output_shape(self):
        layer = Reorg(2)
        testx = torch.rand(2, 3, 224, 224)

        output = layer(testx)
        assert_equal(output.shape, torch.Size([2, 12, 112, 112]))


class Test_bottleneck():

    def test_output_shape(self):
        """ input: 4 
            block: 8 -> 4 -> 8
         """
        layer = bottleneck(8, 3)
        testx = torch.rand(2, 4, 224, 224)

        output = layer(testx)
        assert_equal(output.shape, torch.Size([2, 8, 224, 224]))


class Test_convblock():

    def test_1x1_output_shape(self):
        layer = convblock(3, 16, 1)
        testx = torch.rand(2, 3, 224, 224)

        output = layer(testx)
        assert_equal(output.shape, torch.Size([2, 16, 224, 224]))

    def test_3x3_output_shape(self):
        layer = convblock(3, 16, 3)
        testx = torch.rand(2, 3, 224, 224)

        output = layer(testx)
        assert_equal(output.shape, torch.Size([2, 16, 224, 224]))
