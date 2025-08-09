import pytest
import torch

from ...models import MultiKernelConvModule


class TestMultiKernelConvModule:
    @pytest.mark.parametrize(
        "dimension,x",
        [
            ["1d", torch.rand(8, 3, 32)],
            ["2d", torch.rand(8, 3, 32, 32)],
            ["3d", torch.rand(8, 3, 32, 32, 32)],
        ],
    )
    def test_output_shape(self, dimension: str, x: torch.Tensor):
        model = MultiKernelConvModule(
            3,
            10,
            kernels=[3, 5, 7],
            dimension=dimension,
            concat_output=True,
        )
        y = model(x)

        assert y.shape == torch.Size((8, 3 * 10, *x.shape[2:]))

    def test_reduction_none_output_shape(self):
        x = torch.rand(8, 3, 32, 32)
        model = MultiKernelConvModule(3, 10, kernels=[3, 5, 7], reduction="none")
        y = model(x)

        assert len(y) == 3
        for ele in y:
            assert ele.shape == torch.Size((8, 10, 32, 32))

    @pytest.mark.parametrize("reduction", ["max", "mean"])
    def test_reduction_output_shape(self, reduction: str):
        x = torch.rand(8, 3, 32, 32)
        model = MultiKernelConvModule(3, 10, kernels=[3, 5, 7], reduction=reduction)
        y = model(x)

        assert len(y) == 3
        for ele in y:
            assert ele.shape == torch.Size((8, 10))

    def test_disable_padding_output_shape(self):
        x = torch.rand(8, 3, 32, 32)
        model = MultiKernelConvModule(
            3,
            10,
            kernels=[3, 5, 7],
            disable_padding=True,
        )
        y = model(x)

        assert len(y) == 3
        for ele, k in zip(y, [3, 5, 7]):
            assert ele.shape == torch.Size((8, 10, 32 - k + 1, 32 - k + 1))
