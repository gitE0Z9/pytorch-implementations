import torch
from torch.testing import assert_close

from ..utils.train import build_heatmap


def test_gaussian_heatmap():
    x = torch.Tensor(
        [
            [
                [1, 1],
                [1, 2],
            ]
        ],
    )
    y = build_heatmap(x, (3, 3), sigma=1)

    assert_close(y.shape, torch.Size((1, 2, 3, 3)))
    assert y[0, 0, 1, 1] == y[0, 0].max()
    assert y[0, 1, 1, 2] == y[0, 1].max()
