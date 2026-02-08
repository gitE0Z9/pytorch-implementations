import pytest
import torch


from ..models.deeplabv2.network import deeplab_v2_style_resnet
from ..models.deeplabv2.model import DeepLabV2
from ..models.scale_aware.loss import ScaleAwareLoss
from ..models.scale_aware.model import ScaleAware
from ..models.scale_aware.network import ScaleAwareAttention

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 224
DOWNSCALE_IMAGE_SIZE = IMAGE_SIZE // 8
FEATURE_MAP_SIZE = IMAGE_SIZE // 32
NUM_CLASS = 21


class TestNetwork:
    def test_scale_aware_attention_forward_shape(self):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)

        model = ScaleAwareAttention(INPUT_CHANNEL, num_scales=2)

        y = model((x,))

        assert y.shape == torch.Size(
            (BATCH_SIZE, 2, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )


class TestModel:
    @pytest.mark.parametrize("is_training", (True, False))
    @pytest.mark.parametrize("output_attention", (True, False))
    def test_scale_aware_forward_shape(self, is_training: bool, output_attention: bool):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        m = DeepLabV2(backbone, output_size=NUM_CLASS, enable_shallow_aspp=True)
        model = ScaleAware(m, scales=(1, 0.5))

        if is_training:
            model.train()
        else:
            model.eval()

        y = model(x, output_attention=output_attention)
        if output_attention:
            y, a = y

        if is_training:
            for ele in y:
                assert ele.shape == torch.Size(
                    (BATCH_SIZE, NUM_CLASS, IMAGE_SIZE // 8, IMAGE_SIZE // 8)
                )

        else:
            assert y.shape == torch.Size(
                (BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE)
            )

        if output_attention:
            if is_training:
                assert a.shape == torch.Size(
                    (BATCH_SIZE, 2, IMAGE_SIZE // 8, IMAGE_SIZE // 8)
                )
            else:
                assert a.shape == torch.Size((BATCH_SIZE, 2, IMAGE_SIZE, IMAGE_SIZE))

    def test_scale_aware_backward(self):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        y = torch.randint(
            0, NUM_CLASS, (BATCH_SIZE, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        m = DeepLabV2(backbone, output_size=NUM_CLASS, enable_shallow_aspp=True)
        model = ScaleAware(m, scales=(1, 0.5))
        model.train()

        criterion = ScaleAwareLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)


class TestLoss:
    def test_scale_aware_loss_forward_shape(self):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        y = torch.randint(
            0, NUM_CLASS, (BATCH_SIZE, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        m = DeepLabV2(backbone, output_size=NUM_CLASS, enable_shallow_aspp=True)
        model = ScaleAware(m, scales=(1, 0.5))
        model.train()

        criterion = ScaleAwareLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        assert not torch.isnan(loss)

    def test_scale_aware_loss_backward(self):
        x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        y = torch.randint(
            0, NUM_CLASS, (BATCH_SIZE, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
        )

        backbone = deeplab_v2_style_resnet("resnet50", trainable=False)
        m = DeepLabV2(backbone, output_size=NUM_CLASS, enable_shallow_aspp=True)
        model = ScaleAware(m, scales=(1, 0.5))
        model.train()

        criterion = ScaleAwareLoss()

        yhat = model(x)
        loss = criterion(yhat, y)

        loss.backward()
