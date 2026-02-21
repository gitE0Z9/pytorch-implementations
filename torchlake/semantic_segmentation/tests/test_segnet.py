import pytest
import torch
from torch import nn

from torchlake.common.models.vgg_feature_extractor import VGGFeatureExtractor

from ..models.segnet.model import SegNet, BayesianSegNet
from ..models.segnet.network import DecoderBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 320
FEATURE_MAP_SIZE = IMAGE_SIZE // 32
NUM_CLASS = 21


class TestNetwork:
    def test_decoder_block_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))
        pooling_indices = torch.randint(
            0,
            FEATURE_MAP_SIZE**2,
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE),
        )
        model = DecoderBlock(HIDDEN_DIM, HIDDEN_DIM + 2, 3)

        y = model(x, pooling_indices)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM + 2, FEATURE_MAP_SIZE * 2, FEATURE_MAP_SIZE * 2)
        )


class TestModel:
    def test_segnet_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = VGGFeatureExtractor(
            "vgg16",
            layer_type="maxpool",
            trainable=False,
            enable_bn=True,
            return_pooling_indices=True,
        )
        backbone.fix_target_layers(("5_1",))
        model = SegNet(backbone, NUM_CLASS)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))

    @pytest.mark.parametrize("is_training", (True, False))
    @pytest.mark.parametrize("output_uncertainty", (True, False))
    def test_bayesian_segnet_forward_shape(
        self, is_training: bool, output_uncertainty: bool
    ):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = VGGFeatureExtractor(
            "vgg16",
            layer_type="maxpool",
            trainable=False,
            enable_bn=True,
            return_pooling_indices=True,
        )
        backbone.fix_target_layers(("5_1",))
        model = BayesianSegNet(backbone, NUM_CLASS)
        if is_training:
            model.train()
        else:
            model.eval()

        y = model(x, output_uncertainty=output_uncertainty)
        if output_uncertainty and not is_training:
            y, uncertainty = y
            assert uncertainty.shape == torch.Size(
                (BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE)
            )

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
