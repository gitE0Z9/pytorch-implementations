import torch
from torch import nn

from torchlake.common.models.vgg_feature_extractor import VGGFeatureExtractor

from ..models.segnet.model import SegNet
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
