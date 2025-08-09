import pytest
import torch
from torch import nn
from torch.testing import assert_close

from ...models import ChannelShuffle


class TestChannelShuffle:
    @pytest.mark.parametrize("groups", [1, 2, 3, 4, 8])
    def test_output_shape(self, groups: int):
        x = torch.randn(2, 48, 224, 224)
        layer = ChannelShuffle(groups=groups)
        official_layer = nn.ChannelShuffle(groups)
        y, official_y = layer(x), official_layer(x)

        assert_close(y, official_y)
