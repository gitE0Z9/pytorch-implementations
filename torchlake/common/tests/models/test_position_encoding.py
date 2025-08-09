import pytest
import torch
from torch.testing import assert_close

from ...models import PositionEncoding1d


class TestPositionEncoding:
    @pytest.mark.parametrize("trainable", [True, False])
    def test_1d_output_shape(self, trainable: bool):
        s, h = 32, 16
        x = torch.rand(2, s, h)

        model = PositionEncoding1d(s, h, trainable)
        y = model.forward(x)

        assert_close(y.shape, torch.Size((1, s, h)))
