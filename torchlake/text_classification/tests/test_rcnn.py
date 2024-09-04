import torch

from ..models import Rcnn


def test_rcnn_output_shape():
    """test output shape"""
    model = Rcnn(100, 8, 8, 10)

    x = torch.randint(0, 100, (4, 256))
    output = model.forward(x)

    assert output.shape == torch.Size((4, 10))
