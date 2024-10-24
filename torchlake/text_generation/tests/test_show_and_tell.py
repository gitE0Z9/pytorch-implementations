from functools import partial

import pytest
import torch
from torchlake.common.models import VGGFeatureExtractor

from ..models.show_and_tell import NeuralImageCation
from .constants import BATCH_SIZE, CONTEXT, HIDDEN_DIM, SEQ_LEN, VOCAB_SIZE


@pytest.mark.parametrize(
    "is_train,expected_shape",
    [(True, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)), (False, (BATCH_SIZE, SEQ_LEN))],
)
def test_forward_shape(is_train: bool, expected_shape: tuple[int]):
    x = torch.rand((BATCH_SIZE, 3, 224, 224))
    y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    extractor = VGGFeatureExtractor("vgg16", "maxpool", trainable=False)
    extractor.forward = partial(extractor.forward, target_layer_names=["5_1"])
    model = NeuralImageCation(
        extractor,
        VOCAB_SIZE,
        512,
        HIDDEN_DIM,
        VOCAB_SIZE,
        context=CONTEXT,
    )

    if is_train:
        model.train()
        y = model(x, y)
        assert y.shape == torch.Size(expected_shape)
    else:
        model.eval()
        y = model(x)
        assert y.size(0) == BATCH_SIZE
        assert y.size(1) <= SEQ_LEN + 1
