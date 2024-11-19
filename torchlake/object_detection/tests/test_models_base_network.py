import torch
from torch.testing import assert_close

from ..models.base.network import RegHead


class TestRegHead:
    def test_output_shape(self):
        model = RegHead(16, 20, 5, 4)
        test_x = torch.rand(2, 16, 7, 7)

        coord, conf = model(test_x)
        assert_close(coord.shape, torch.Size([2, 4 * 5, 7, 7]))
        assert_close(conf.shape, torch.Size([2, 20 * 5, 7, 7]))
