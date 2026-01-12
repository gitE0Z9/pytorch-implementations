import pytest
import torch

from ..models.deeplabv3.model import DeepLabV3
from ..models.deeplabv3.network import CascadeASPP, ASPP
from ..models.deeplabv2.network import deeplab_v2_style_resnet

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 321
DOWNSCALE_IMAGE_SIZE = IMAGE_SIZE // 8 + 1
NUM_CLASS = 21


class TestNetwork:
    def test_aspp_forward_shape(self):
        x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

        model = ASPP(HIDDEN_DIM, HIDDEN_DIM, [6, 12])
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

    def test_cascade_aspp_forward_shape(self):
        x = torch.rand(
            (BATCH_SIZE, HIDDEN_DIM, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

        model = CascadeASPP(HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM)
        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("is_train", [True, False])
    @pytest.mark.parametrize(
        "neck_type,dilation_size_32x", [("parallel", 1), ("cascade", 4)]
    )
    def test_deeplab_v3_forward_shape(
        self,
        is_train: bool,
        neck_type: str,
        dilation_size_32x: int,
    ):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet(
            "resnet50",
            trainable=False,
            dilation_size_16x=2,
            dilation_size_32x=dilation_size_32x,
        )
        model = DeepLabV3(backbone, output_size=NUM_CLASS, neck_type=neck_type)
        if is_train:
            model.train()
        else:
            model.eval()

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))

    def test_deeplab_v3_backward(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        y = torch.randint(0, NUM_CLASS, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE))

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        model = DeepLabV3(backbone, output_size=NUM_CLASS)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)
