import pytest
import torch

from ..models.parsenet import ParseNetDeepLab, ParseNetFCN, GlobalContextModule


class TestParseNetFCN:
    @pytest.mark.parametrize("num_skip_connection", [0, 1, 2])
    def test_forward_shape(self, num_skip_connection: int):
        x = torch.rand((1, 3, 32, 32))

        model = ParseNetFCN(
            21,
            num_skip_connection=num_skip_connection,
        )

        y = model(x)

        assert y.shape == torch.Size((1, 21, 32, 32))


class TestParseNetDeepLab:
    @pytest.mark.parametrize(
        "is_train,expected",
        [
            [True, 321 // 8 + 1],
            [False, 321],
        ],
    )
    def test_forward_shape(self, is_train: bool, expected: int):
        x = torch.rand((1, 3, 321, 321))

        model = ParseNetDeepLab(output_size=21)
        if is_train:
            model.train()
        else:
            model.eval()

        y = model(x)

        assert y.shape == torch.Size((1, 21, expected, expected))

    def test_backward(self):
        x = torch.rand((1, 3, 321, 321))
        y = torch.randint(0, 21, (1, 41, 41))

        model = ParseNetDeepLab(output_size=21)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)


class TestGlobalContextModule:
    def test_forward_shape(self):
        x = torch.rand((1, 3, 32, 32))

        model = GlobalContextModule(3)

        y = model(x)

        assert y.shape == torch.Size((1, 6, 32, 32))
