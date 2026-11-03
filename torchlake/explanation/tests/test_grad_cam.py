import pytest

import torch
from torchvision.models.resnet import resnet18
from ..models.grad_cam.model import GradCAM

BATCH_SIZE = 4
INPUT_CHANNEL = 3
IMAGE_SIZE = 224


class TestModel:
    @pytest.mark.parametrize("layer_name", ("layer3", "layer4"))
    @pytest.mark.parametrize(
        "target",
        (
            torch.randint(
                0,
                1000,
                (BATCH_SIZE,),
            ),
            None,
        ),
    )
    def test_grad_cam_forward_shape(self, layer_name, target):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        m = resnet18()
        model = GradCAM(m)
        model.set_activation_layer(layer_name)

        y = model(x, target=target)

        assert y.shape == (BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)
