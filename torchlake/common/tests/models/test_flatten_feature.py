from math import prod

import pytest
import torch

from ...models import FlattenFeature


class TestFlattenFeature:
    @pytest.mark.parametrize(
        "input_shape,dimension",
        [[(7,), "1d"], [(7, 7), "2d"], [(7, 7, 7), "3d"]],
    )
    @pytest.mark.parametrize("start_dim", [1, 2])
    @pytest.mark.parametrize("reduction", ["mean", "max", None])
    def test_output_shape(
        self,
        input_shape: tuple[int],
        dimension: str,
        start_dim: int,
        reduction: str,
    ):
        x = torch.randn(8, 32, *input_shape)

        model = FlattenFeature(reduction, dimension, start_dim)

        y = model(x)

        reduced_factor = 1 if reduction is not None else prod(input_shape)
        if start_dim == 1:
            expected_shape = (8, 32 * reduced_factor)
        else:
            expected_shape = (8, 32, reduced_factor)
        assert y.shape == torch.Size(expected_shape)
