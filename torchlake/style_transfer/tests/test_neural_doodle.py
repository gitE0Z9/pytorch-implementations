import torch
from torchlake.common.models import VGGFeatureExtractor

from ..models.neural_doodle import MRFLoss, NeuralDoodle


class TestModel:
    def test_forward_shape(self):
        style_layers = ["3_1", "4_1"]
        _input = torch.rand((1, 3, 224, 224))
        mask = torch.rand((1, 3, 224, 224))

        extractor = VGGFeatureExtractor("vgg19", "relu", False)
        model = NeuralDoodle(extractor, style_layers)

        features = model(_input, mask)

        assert len(features) == len(style_layers)
        assert features[0].shape == torch.Size((1, extractor.feature_dims[2], 56, 56))
        assert features[1].shape == torch.Size((1, extractor.feature_dims[3], 28, 28))


class TestLoss:
    def test_forward(self):
        _input = torch.rand((1, 3, 32, 32))
        mask = torch.rand((1, 3, 32, 32))
        criterion = MRFLoss(1, 1)

        loss, content_score, style_score, tv_score = criterion(_input, mask)
        assert not torch.isnan(loss)
        assert content_score == 0
        assert not torch.isnan(style_score)
        assert tv_score == 0

    def test_backward(self):
        _input = torch.rand((1, 3, 32, 32)).requires_grad_()
        mask = torch.rand((1, 3, 32, 32))
        criterion = MRFLoss(1, 1)

        loss, _, _, _ = criterion(_input, mask)
        loss.backward()
