import torch

from ..models.xception import Xception


class TestXception:
    def test_output_shape(self):
        x = torch.randn(2, 3, 299, 299)

        model = Xception(output_size=16)

        y = model(x)

        assert y.shape == torch.Size((2, 16))
