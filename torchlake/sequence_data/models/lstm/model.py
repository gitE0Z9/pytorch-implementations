from torch import nn

from ..base.wrapper import SequenceModelWrapper


class LSTMClassifier(SequenceModelWrapper):
    def __init__(self, *args, **kwargs):
        super(LSTMClassifier, self).__init__(model_class=nn.LSTM, *args, **kwargs)
