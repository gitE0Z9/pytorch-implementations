from torch import nn

from ..base.wrapper import SequenceModelWrapper


class RnnClassifier(SequenceModelWrapper):
    def __init__(self, *args, **kwargs):
        super(RnnClassifier, self).__init__(model_class=nn.RNN, *args, **kwargs)
