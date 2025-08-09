from math import ceil

import pytest
import torch
from torchvision.ops import Conv2dNormActivation

from ...models import ResBlock


@pytest.mark.parametrize(
    "name,stride",
    [
        ["stride=1", 1],
        ["stride=2", 2],
    ],
)
class TestResBlock:
    def test_output_shape(self, name: str, stride: int):
        INPUT_SHAPE = 7
        OUTPUT_SHAPE = ceil(7 / stride)
        x = torch.randn(8, 16, INPUT_SHAPE, INPUT_SHAPE)

        model = ResBlock(
            16,
            32,
            Conv2dNormActivation(
                16,
                32,
                3,
                stride,
                activation_layer=None,
            ),
            stride,
        )

        y = model(x)

        assert y.shape == torch.Size((8, 32, OUTPUT_SHAPE, OUTPUT_SHAPE))
