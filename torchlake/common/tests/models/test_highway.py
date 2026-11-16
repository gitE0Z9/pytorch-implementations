from unittest import TestCase

import torch
from torchvision.ops import Conv2dNormActivation

from ...models import HighwayLayer


class TestHighwayLayer(TestCase):
    def test_output_shape(self):
        x = torch.randn(8, 32, 7, 7)

        model = HighwayLayer(
            Conv2dNormActivation(32, 32, 3),
            Conv2dNormActivation(32, 32, 3),
        )

        y = model(x)

        assert y.shape == torch.Size((8, 32, 7, 7))
