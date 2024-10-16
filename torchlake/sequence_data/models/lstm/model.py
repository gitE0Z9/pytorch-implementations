from torch import nn

from ..base.rnn_discriminator import RNNDiscriminator
from ..base.decorators import register_model_class


@register_model_class(nn.LSTM)
class LSTMDiscriminator(RNNDiscriminator): ...
