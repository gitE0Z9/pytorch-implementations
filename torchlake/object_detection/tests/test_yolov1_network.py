import torch
from torch.testing import assert_close

from ..models.yolov1.network import Extraction


class TestExtraction:
    def test_output_shape(self):
        model = Extraction()
        test_x = torch.rand(2, 3, 224, 224)

        output: torch.Tensor = model(test_x)
        assert_close(output.shape, torch.Size([2, 1000]))
