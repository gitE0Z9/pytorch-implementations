import torch
from torchlake.common.models import VGGFeatureExtractor

from ..models.adain.loss import AdaInLoss
from ..models.adain.model import AdaInTrainer
from ..models.adain.network import AdaIn2d, AdaInDecoder


def test_forward_shape_layer():
    c, s = torch.rand((2, 3, 256, 256)), torch.rand((2, 3, 256, 256))
    layer = AdaIn2d(c, s)
    output = layer(c)

    assert output.shape == torch.Size((2, 3, 256, 256))


def test_forward_shape_decoder():
    x = torch.rand((2, 512, 32, 32))
    decoder = AdaInDecoder()
    output = decoder(x)

    assert output.shape == torch.Size((2, 3, 256, 256))


def test_forward_shape_model():
    c, s = torch.rand((2, 3, 256, 256)), torch.rand((2, 3, 256, 256))

    STYLE_LAYER_NAMES = ["1_1", "2_1", "3_1", "4_1"]
    feature_extractor = VGGFeatureExtractor("vgg19", "relu")
    model = AdaInTrainer(feature_extractor, STYLE_LAYER_NAMES)
    output = model(c, s)

    assert output.shape == torch.Size((2, 3, 256, 256))


def test_style_interpolation_forward_shape_model():
    c, s = torch.rand((2, 3, 256, 256)), torch.rand((2, 3, 256, 256))

    STYLE_LAYER_NAMES = ["1_1", "2_1", "3_1", "4_1"]
    feature_extractor = VGGFeatureExtractor("vgg19", "relu")
    model = AdaInTrainer(feature_extractor, STYLE_LAYER_NAMES)
    output = model.style_interpolation_forward(c, s, [0.5, 0.5])

    assert output.shape == torch.Size((2, 3, 256, 256))


def test_loss_forward_shape_model():
    c, s = torch.rand((2, 3, 256, 256)), torch.rand((2, 3, 256, 256))

    STYLE_LAYER_NAMES = ["1_1", "2_1", "3_1", "4_1"]
    feature_extractor = VGGFeatureExtractor("vgg19", "relu")
    model = AdaInTrainer(feature_extractor, STYLE_LAYER_NAMES)
    generated_features, normalized, style_features = model.loss_forward(c, s)

    assert len(generated_features) == len(STYLE_LAYER_NAMES)
    assert len(style_features) == len(STYLE_LAYER_NAMES)
    assert normalized.shape == torch.Size((2, 512, 32, 32))

    for i, generated_feature in enumerate(generated_features):
        i = 2**i
        l = 64 * i
        assert generated_feature.shape == torch.Size((2, l, 256 // i, 256 // i))

    for i, style_feature in enumerate(style_features):
        i = 2**i
        l = 64 * i
        assert style_feature.shape == torch.Size((2, l, 256 // i, 256 // i))


def test_forward_shape_loss():
    c, s = torch.rand((2, 3, 256, 256)), torch.rand((2, 3, 256, 256))

    STYLE_LAYER_NAMES = ["1_1", "2_1", "3_1", "4_1"]
    feature_extractor = VGGFeatureExtractor("vgg19", "relu")
    model = AdaInTrainer(feature_extractor, STYLE_LAYER_NAMES)
    generated_features, normalized, style_features = model.loss_forward(c, s)

    criterion = AdaInLoss()
    total_loss, content_loss, style_loss = criterion(
        generated_features,
        normalized,
        style_features,
    )

    assert not torch.isnan(total_loss).any()
    assert not torch.isnan(content_loss).any()
    assert not torch.isnan(style_loss).any()
