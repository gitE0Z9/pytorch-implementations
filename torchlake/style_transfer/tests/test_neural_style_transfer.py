import torch

from torchlake.common.models import VGGFeatureExtractor

from ..models.neural_style_transfer import NeuralStyleTransferLoss

BATCH_SIZE = 1
INPUT_CHANNEL = 3
IMAGE_SIZE = 224
CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1
CONTENT_LAYER_NAME = "3_1"
STYLE_LAYER_NAMES = ["1_1", "2_1", "3_1", "4_1", "5_1"]


class TestLoss:
    def setup_neural_style_transfer_loss(self):
        content = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        style = torch.rand((BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE))
        output = torch.rand(
            (BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE), requires_grad=True
        )

        backbone = VGGFeatureExtractor("vgg16", "relu", trainable=False)
        criterion = NeuralStyleTransferLoss(
            backbone,
            CONTENT_LAYER_NAME,
            STYLE_LAYER_NAMES,
            CONTENT_WEIGHT,
            STYLE_WEIGHT,
            return_all_loss=True,
        )
        criterion.set_style_features(style)

        return criterion(output, content)

    def test_neural_style_transfer_loss_forward(self):
        loss, content_score, style_score = self.setup_neural_style_transfer_loss()

        assert not torch.isnan(loss)
        assert not torch.isnan(content_score)
        assert not torch.isnan(style_score)

    def test_neural_style_transfer_loss_backward(self):
        loss, _, _ = self.setup_neural_style_transfer_loss()

        loss.backward()
