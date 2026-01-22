import pytest
import torch

from ..models.r_aspp.model import MobileNetV2Seg
from ..models.r_aspp.network import r_aspp_style_mobilenet

BATCH_SIZE = 2
INPUT_CHANNEL = 3
IMAGE_SIZE = 321
NUM_CLASS = 21


class TestNetwork:
    @pytest.mark.parametrize(
        "network_name", ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]
    )
    def test_r_aspp_style_mobilenet_forward_shape(self, network_name: str):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        model = r_aspp_style_mobilenet(network_name, trainable=False)
        features = model(x, ("0_1", "1_1", "2_1", "3_1", "4_1", "output"))

        dims = (
            model.hidden_dim_2x[0],
            model.hidden_dim_4x[0],
            model.hidden_dim_8x[0],
            model.hidden_dim_16x[0],
            model.hidden_dim_32x[0],
            model.feature_dim,
        )
        for f, d in zip(features, dims):
            assert f.shape[:2] == torch.Size((BATCH_SIZE, d))


class TestModel:
    @pytest.mark.parametrize("is_train", [True, False])
    def test_mobilenet_v2_seg_forward_shape(self, is_train: bool):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = r_aspp_style_mobilenet("mobilenet_v2", trainable=False)
        model = MobileNetV2Seg(backbone, NUM_CLASS)
        if is_train:
            model.train()
        else:
            model.eval()

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))

    def test_mobilenet_v2_seg_backward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        backbone = r_aspp_style_mobilenet("mobilenet_v2", trainable=False)
        model = MobileNetV2Seg(backbone, NUM_CLASS)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)
