import pytest
import torch

from ..models.mscad import MSCAD, ContextModule


class TestMSCAD:
    @pytest.mark.parametrize("context_type", ["basic", "large"])
    def test_forward_shape(self, context_type: str):
        x = torch.rand((2, 3, 320, 320))

        model = MSCAD(21, context_type)
        y = model(x)

        assert y.shape == torch.Size((2, 21, 320, 320))


class TestContextModule:
    def test_forward_shape(self):
        x = torch.rand((2, 32, 7, 7))

        model = ContextModule(32, [1] * 8)
        y = model.forward(x)

        assert y.shape == torch.Size((2, 32, 7, 7))
