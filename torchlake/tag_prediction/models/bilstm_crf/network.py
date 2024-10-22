import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext

from ...utils.decode import viterbi_decode


class LinearCRF(nn.Module):
    def __init__(self, output_size: int, context: NlpContext = NlpContext()):
        """CRF(conditional random field) layer

        Args:
            output_size (int): size of output
            context (NlpContext, optional): nlp context. Defaults to NlpContext().
        """
        super().__init__()
        self.context = context

        # transition matrix
        # store not prob but log likelihood
        # from i to j
        self.transition = nn.Parameter(torch.rand(output_size, output_size))

        MUST_NOT = 0
        MUST_HAPPEN = 1

        # prohibit transition to bos
        self.transition.data[:, context.bos_idx] = MUST_NOT
        # must transfer from the start tag
        self.transition.data[context.bos_idx, :] = MUST_HAPPEN
        # never transfer from the start to the end
        self.transition.data[context.eos_idx, context.bos_idx] = MUST_NOT
        # never transfer from the eos
        self.transition.data[context.eos_idx, :] = MUST_NOT
        # never transfer from the start to pad
        self.transition.data[context.bos_idx, context.padding_idx] = MUST_NOT
        # must absorb into pad
        self.transition.data[context.eos_idx, context.padding_idx] = MUST_HAPPEN
        # must absorb into pad
        self.transition.data[context.padding_idx, context.padding_idx] = MUST_HAPPEN

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        output_score: bool = False,
    ) -> tuple[torch.Tensor] | torch.Tensor:
        """forward, use viterbi decode to find best path

        Args:
            x (torch.Tensor): prediction probability, shape is (batch_size, sequence_length, label_size)
            mask (torch.Tensor | None, optional): mask for padding index, shape is (batch_size, sequence_length). Defaults to None.
            output_score (bool, optional): return score of viterbi path. Defaults to False.

        Returns:
            tuple[torch.Tensor] | torch.Tensor: crf paths and score if `output_score` is `True`
        """
        path, score = viterbi_decode(x, self.transition, mask, self.context)

        if output_score:
            return path, score
        else:
            return path
