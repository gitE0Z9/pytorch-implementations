from torch import nn

from ..base.wrapper import SequenceModelWrapper


class LstmClassifier(SequenceModelWrapper):
    def __init__(self, *args, **kwargs):
        super(LstmClassifier, self).__init__(model_class=nn.LSTM, *args, **kwargs)
