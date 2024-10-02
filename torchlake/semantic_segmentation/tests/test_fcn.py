import pytest
import torch

from ..models.fcn.model import FCN, FCNLegacy


@pytest.mark.parametrize("num_skip_connection", [1, 2, 3])
def test_legacy_forward_shape(num_skip_connection: int):
    x = torch.rand((1, 3, 32, 32))

    model = FCNLegacy(
        21,
        num_skip_connection=num_skip_connection,
    )

    y = model(x)

    assert y.shape == torch.Size((1, 21, 32, 32))


@pytest.mark.parametrize("num_skip_connection", [1, 2, 3])
def test_forward_shape(num_skip_connection: int):
    x = torch.rand((1, 3, 32, 32))

    model = FCN(
        21,
        num_skip_connection=num_skip_connection,
    )

    y = model(x)

    assert y.shape == torch.Size((1, 21, 32, 32))
