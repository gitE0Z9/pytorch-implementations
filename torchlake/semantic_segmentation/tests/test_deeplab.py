import pytest
import torch

from ..models import DeepLab


@pytest.mark.parametrize(
    "is_train,expected",
    [
        [True, 321 // 8 + 1],
        [False, 321],
    ],
)
def test_forward_shape(is_train: bool, expected: int):
    x = torch.rand((1, 3, 321, 321))

    model = DeepLab(output_size=21)
    if is_train:
        model.train()
    else:
        model.eval()

    y = model(x)

    assert y.shape == torch.Size((1, 21, expected, expected))


def test_backward():
    x = torch.rand((1, 3, 321, 321))
    y = torch.randint(0, 21, (1, 41, 41))

    model = DeepLab(output_size=21)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    yhat = model(x)
    loss = criterion(yhat, y)

    loss.backward()

    assert not torch.isnan(loss)
