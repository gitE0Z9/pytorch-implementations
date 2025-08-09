import torch
from torch.testing import assert_close

from ...models import StackedPatch2d


class TestStackedPatch:
    def test_forward_shape_2d(self):
        layer = StackedPatch2d(2)
        test_x = torch.rand(2, 3, 224, 224)

        output: torch.Tensor = layer(test_x)
        assert_close(output.shape, torch.Size([2, 12, 112, 112]))
