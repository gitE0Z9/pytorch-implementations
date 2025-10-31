import torch

from ..models.crnn.model import CRNN


class TestModel:
    def test_crnn_forward_shape(self):
        x = torch.rand(1, 3, 32, 1698)
        model = CRNN(3, 8, 10)
        output = model(x)

        assert output.shape == torch.Size((1698 // 4 - 1, 1, 10))
