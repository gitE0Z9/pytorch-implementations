from math import floor

import torch

from ..controllers.trainer_deeplabv2 import DeepLabV2Trainer
from ..models.deeplabv2 import DeepLabV2
from ..models.deeplabv2.network import deeplab_v2_style_vgg

BATCH_SIZE = 2
INPUT_CHANNEL = 3
HIDDEN_DIM = 8
IMAGE_SIZE = 321
DOWNSCALE_IMAGE_SIZE = IMAGE_SIZE // 8 + 1
NUM_CLASS = 21


def test_predict_output_shape():
    x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

    backbone = deeplab_v2_style_vgg("vgg16", trainable=False)
    model = DeepLabV2(backbone, output_size=NUM_CLASS)

    trainer = DeepLabV2Trainer(1, "cpu")
    trainer.set_multiscales()

    output = trainer._predict((x, None), model)

    for ele, scale in zip(
        output,
        (
            DOWNSCALE_IMAGE_SIZE,
            floor(IMAGE_SIZE * 0.75) // 8 + 1,
            floor(IMAGE_SIZE * 0.5) // 8 + 1,
        ),
    ):
        assert ele.shape == torch.Size((BATCH_SIZE, NUM_CLASS, scale, scale))


def test_calc_loss_output_shape():
    x = torch.rand(BATCH_SIZE, INPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
    y = torch.randint(
        0, NUM_CLASS, (BATCH_SIZE, DOWNSCALE_IMAGE_SIZE, DOWNSCALE_IMAGE_SIZE)
    ).float()

    backbone = deeplab_v2_style_vgg("vgg16", trainable=False)
    model = DeepLabV2(backbone, output_size=NUM_CLASS)

    trainer = DeepLabV2Trainer(1, "cpu")
    trainer.set_multiscales()
    criterion = torch.nn.CrossEntropyLoss()

    output = trainer._predict((x, y), model)
    loss = trainer._calc_loss(output, (x, y), criterion)

    assert not torch.isnan(loss)
