import pytest
import torch

from ...models import PositionEncoding1d

BATCH_SIZE = 2
SEQ_LEN = 32
HIDDEN_DIM = 16


class TestPositionEncoding:
    @pytest.mark.parametrize(
        "is_fixed,trainable",
        [
            (True, True),
            (True, False),
            (False, False),
            pytest.param(
                False, True, marks=pytest.mark.xfail(raises=AssertionError, strict=True)
            ),
        ],
    )
    def test_1d_output_shape(self, is_fixed: bool, trainable: bool):
        x = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

        model = PositionEncoding1d(
            SEQ_LEN,
            HIDDEN_DIM,
            is_fixed=is_fixed,
            trainable=trainable,
        )

        y = model(x)

        assert y.shape == torch.Size((1, SEQ_LEN, HIDDEN_DIM))
