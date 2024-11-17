import torch
from torch.testing import assert_close

from ..models.yolov2.network import ReorgLayer


class Test_Reorg:
    def test_output_shape(self):
        layer = ReorgLayer(2)
        test_x = torch.rand(2, 3, 224, 224)

        output: torch.Tensor = layer(test_x)
        assert_close(output.shape, torch.Size([2, 12, 112, 112]))
