import torch

from ...models import L2Norm


class TestL2Norm:
    def test_output_shape(self):
        x = torch.rand(1, 32, 28, 28)
        model = L2Norm(32)
        y = model.forward(x)

        assert y.shape == x.shape
