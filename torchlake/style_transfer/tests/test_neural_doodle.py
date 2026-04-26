import torch
from torchlake.common.models import VGGFeatureExtractor

from ..models.neural_doodle import MRFLoss, NeuralDoodle

BATCH_SIZE = 1
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
FEATURE_MAP_SIZE = 32


class TestModel:
    def test_neural_doodle_forward_shape(self):
        style_layers = ["3_1", "4_1"]
        _input = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        mask = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        extractor = VGGFeatureExtractor("vgg19", "relu", False)
        model = NeuralDoodle(extractor, style_layers)

        features = model(_input, mask)

        assert len(features) == len(style_layers)
        assert features[0].shape == torch.Size(
            (BATCH_SIZE, extractor.feature_dims[2], IMAGE_SIZE // 4, IMAGE_SIZE // 4)
        )
        assert features[1].shape == torch.Size(
            (BATCH_SIZE, extractor.feature_dims[3], IMAGE_SIZE // 8, IMAGE_SIZE // 8)
        )


class TestLoss:
    def test_mrf_loss_forward(self):
        content = torch.rand(
            (BATCH_SIZE, INPUT_CHANNEL, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )
        mask = torch.rand(
            (BATCH_SIZE, INPUT_CHANNEL, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )
        criterion = MRFLoss(1, 1)

        loss, content_score, style_score, tv_score = criterion(content, mask)
        assert not torch.isnan(loss)
        assert content_score == 0
        assert not torch.isnan(style_score)
        assert tv_score == 0

    def test_mrf_loss_backward(self):
        content = torch.rand(
            (BATCH_SIZE, INPUT_CHANNEL, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        ).requires_grad_()
        mask = torch.rand(
            (BATCH_SIZE, INPUT_CHANNEL, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
        )
        criterion = MRFLoss(1, 1)

        loss, _, _, _ = criterion(content, mask)
        loss.backward()
