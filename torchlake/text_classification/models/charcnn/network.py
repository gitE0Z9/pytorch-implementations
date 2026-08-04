import torch
import torch.nn.functional as F
from torch import nn
from torchlake.common.schemas.nlp import NLPContext


class CharQuantization(nn.Module):
    def __init__(self, char_size: int, context: NLPContext):
        super().__init__()
        self.char_size = char_size
        self.context = context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.one_hot(x, self.char_size)

        # zero out unknown index
        y[self.context.unk_idx] = 0

        return y
