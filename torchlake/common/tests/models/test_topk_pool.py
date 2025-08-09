from unittest import TestCase

import torch

from ...models import TopKMaxPool1d


class TestTopkPool(TestCase):
    def test_max_1d_output_shape(self):
        x = torch.randn(8, 32, 7)

        model = TopKMaxPool1d(3)

        y = model(x)

        assert y.shape == torch.Size((8, 32, 3))
