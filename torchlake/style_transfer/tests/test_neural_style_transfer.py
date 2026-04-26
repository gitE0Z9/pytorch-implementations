import torch
from torchlake.common.models import VGGFeatureExtractor

from ..models.neural_style_transfer import NeuralStyleTransfer, NeuralStyleTransferLoss

BATCH_SIZE = 1
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
CONTENT_LAYER_NAMES = ["3_1"]
STYLE_LAYER_NAMES = ["1_1", "2_1", "3_1", "4_1", "5_1"]


class TestModel:
    def test_neural_style_transfer_forward_shape(self):
        _input = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))

        extractor = VGGFeatureExtractor("vgg19", "relu", trainable=False)
        model = NeuralStyleTransfer(extractor, CONTENT_LAYER_NAMES, STYLE_LAYER_NAMES)

        features = model(_input, "style")

        assert len(features) == len(STYLE_LAYER_NAMES)
        for i, dim in enumerate(extractor.feature_dims):
            scale = 224 // 2**i
            assert features.pop(0).shape == torch.Size((1, dim, scale, scale))


class TestLoss:
    def test_neural_style_transfer_loss_forward(self):
        content = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        style = torch.rand((5, BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        output = torch.rand((5, BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        criterion = NeuralStyleTransferLoss(2, 1, 1, return_all_loss=True)

        loss, content_score, style_score = criterion(content, [*style], [*output])
        assert not torch.isnan(loss)
        assert not torch.isnan(content_score)
        assert not torch.isnan(style_score)

    def test_neural_style_transfer_loss_backward(self):
        content = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        style = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        output = torch.rand(
            (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        ).requires_grad_()
        criterion = NeuralStyleTransferLoss(2, 1, 1, return_all_loss=True)

        loss, _, _ = criterion(content, [style] * 5, [output] * 5)
        loss.backward()
