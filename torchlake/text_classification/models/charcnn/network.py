import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext
import torch.nn.functional as F


class CharQuantization(nn.Module):
    def __init__(self, char_size: int, context: NlpContext):
        super(CharQuantization, self).__init__()
        self.char_size = char_size
        self.context = context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.one_hot(x, self.char_size)

        # zero out unknown index
        y[self.context.unk_idx] = 0

        return y
