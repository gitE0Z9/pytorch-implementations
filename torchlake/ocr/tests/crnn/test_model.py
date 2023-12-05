import torch
from ocr.models.crnn import Crnn


def test_model_shape():
    test_x = torch.randn(1, 3, 32, 1698)
    output_shape = Crnn(4, 10)(test_x).shape

    assert output_shape == torch.Size((1698 // 4 - 1, 1, 10))
