from torch import nn

from ..base.wrapper import SequenceModelWrapper


class GruClassifier(SequenceModelWrapper):
    def __init__(self, *args, **kwargs):
        super(GruClassifier, self).__init__(model_class=nn.GRU, *args, **kwargs)
