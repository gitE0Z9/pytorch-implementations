import torch

from torchlake.common.models import ResNetFeatureExtractor

from ..models.linknet.model import LinkNet
from ..models.linknet.network import EncoderBlock, DecoderBlock

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 320
FEATURE_MAP_SIZE = IMAGE_SIZE // 32
NUM_CLASS = 21


class TestNetwork:
    def test_encoder_block_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM + 1, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = EncoderBlock(HIDDEN_DIM + 1, HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE // 2, FEATURE_MAP_SIZE // 2)
        )

    def test_decoder_block_forward_shape(self):
        x = torch.rand((BATCH_SIZE, HIDDEN_DIM + 1, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE))

        model = DecoderBlock(HIDDEN_DIM + 1, HIDDEN_DIM)

        y = model(x)

        assert y.shape == torch.Size(
            (BATCH_SIZE, HIDDEN_DIM, FEATURE_MAP_SIZE * 2, FEATURE_MAP_SIZE * 2)
        )


class TestModel:
    def test_linknet_forward_shape(self):
        x = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        backbone = ResNetFeatureExtractor("resnet18", trainable=False)
        backbone.fix_target_layers(("1_1", "2_1", "3_1", "4_1"))
        model = LinkNet(backbone, NUM_CLASS)
        # model = LinkNet(INPUT_CHANNEL, NUM_CLASS)

        y = model(x)

        assert y.shape == torch.Size((BATCH_SIZE, NUM_CLASS, IMAGE_SIZE, IMAGE_SIZE))
