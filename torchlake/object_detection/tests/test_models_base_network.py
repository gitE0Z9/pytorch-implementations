import torch
from torch.testing import assert_close

from ..models.base.network import ConvBlock


class TestConvBlock:
    def test_output_shape(self):
        model = ConvBlock(3, 64, 3)
        test_x = torch.rand(2, 3, 224, 224)

        output: torch.Tensor = model(test_x)
        assert_close(output.shape, torch.Size([2, 64, 224, 224]))
