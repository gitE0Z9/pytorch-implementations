from functools import partial

import pytest
import torch
from torchlake.common.models import VGGFeatureExtractor
from torchlake.sequence_data.models.base import RNNGenerator
from torchlake.sequence_data.models.lstm import LSTMDiscriminator

from ..models.show_and_tell import NeuralImageCation
from .constants import BATCH_SIZE, CONTEXT, HIDDEN_DIM, SEQ_LEN, VOCAB_SIZE


class TestModel:
    @pytest.mark.parametrize("early_stopping", [True, False])
    def test_forward_shape_train(self, early_stopping: bool):
        x = torch.rand((BATCH_SIZE, 3, 224, 224))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

        extractor = VGGFeatureExtractor("vgg16", "maxpool", trainable=False)
        extractor.forward = partial(extractor.forward, target_layer_names=["5_1"])
        decoder = RNNGenerator(
            LSTMDiscriminator(
                VOCAB_SIZE,
                512,
                HIDDEN_DIM,
                VOCAB_SIZE,
                context=CONTEXT,
            )
        )
        model = NeuralImageCation(extractor, decoder, context=CONTEXT)

        model.train()
        y = model(x, y, early_stopping=early_stopping)
        assert y.size(0) == BATCH_SIZE
        if early_stopping:
            assert y.size(1) <= SEQ_LEN
        else:
            assert y.size(1) == SEQ_LEN
        assert y.size(2) == VOCAB_SIZE

    @pytest.mark.parametrize("topk", [1, 2])
    def test_forward_shape_inference(self, topk: int):
        x = torch.rand((BATCH_SIZE, 3, 224, 224))
        y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

        extractor = VGGFeatureExtractor("vgg16", "maxpool", trainable=False)
        extractor.forward = partial(extractor.forward, target_layer_names=["5_1"])
        decoder = RNNGenerator(
            LSTMDiscriminator(
                VOCAB_SIZE,
                512,
                HIDDEN_DIM,
                VOCAB_SIZE,
                context=CONTEXT,
            )
        )
        model = NeuralImageCation(extractor, decoder, context=CONTEXT)

        model.eval()
        y = model(x, topk=topk)
        assert y.size(0) == BATCH_SIZE
        assert y.size(1) <= SEQ_LEN + 1
