import torch

from ...common.models import VGGFeatureExtractor
from ..controllers.trainer_deeplabv2 import DeepLabV2Trainer
from ..models.deeplabv2 import DeepLabV2


def test_predict_output_shape():
    x = torch.rand(2, 3, 321, 321)

    fe = VGGFeatureExtractor("vgg16", "maxpool", trainable=False)
    model = DeepLabV2(fe, 21)

    trainer = DeepLabV2Trainer(1, "cpu")
    trainer.set_multiscales()

    output = trainer._predict((x, None), model)

    for ele, shape in zip(output, [41, 31, 21]):
        assert ele.shape == torch.Size((2, 21, shape, shape))


def test_calc_loss_output_shape():
    x = torch.rand(2, 3, 321, 321)
    y = torch.rand(2, 41, 41)

    fe = VGGFeatureExtractor("vgg16", "maxpool", trainable=False)
    model = DeepLabV2(fe, 21)

    trainer = DeepLabV2Trainer(1, "cpu")
    trainer.set_multiscales()
    criterion = trainer.get_criterion(20)

    output = trainer._predict((x, y), model)
    loss = trainer._calc_loss(output, (x, y), criterion)

    assert not torch.isnan(loss)
