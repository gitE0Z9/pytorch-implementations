import torch
from torchlake.common.models import VGGFeatureExtractor

from ..models.neural_style_transfer import NeuralStyleTransfer, NeuralStyleTransferLoss

CONTENT_LAYER_NAMES = ["3_1"]
STYLE_LAYER_NAMES = ["1_1", "2_1", "3_1", "4_1", "5_1"]


class TestModel:
    def test_forward_shape(self):
        _input = torch.rand((1, 3, 224, 224))

        extractor = VGGFeatureExtractor("vgg19", "relu", trainable=False)
        model = NeuralStyleTransfer(extractor, CONTENT_LAYER_NAMES, STYLE_LAYER_NAMES)

        features = model(_input, "style")

        assert len(features) == len(STYLE_LAYER_NAMES)
        for i, dim in enumerate(extractor.feature_dims):
            scale = 224 // 2**i
            assert features.pop(0).shape == torch.Size((1, dim, scale, scale))


class TestLoss:
    def test_forward(self):
        content = torch.rand((1, 3, 32, 32))
        style = torch.rand((1, 3, 32, 32))
        output = torch.rand((1, 3, 32, 32))
        criterion = NeuralStyleTransferLoss(2, 1, 1)

        loss, content_score, style_score = criterion(content, [style] * 5, [output] * 5)
        assert not torch.isnan(loss)
        assert not torch.isnan(content_score)
        assert not torch.isnan(style_score)

    def test_backward(self):
        content = torch.rand((1, 3, 32, 32))
        style = torch.rand((1, 3, 32, 32))
        output = torch.rand((1, 3, 32, 32)).requires_grad_()
        criterion = NeuralStyleTransferLoss(2, 1, 1)

        loss, _, _ = criterion(content, [style] * 5, [output] * 5)
        loss.backward()
