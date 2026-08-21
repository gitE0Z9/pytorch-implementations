from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from torchlake.common.schemas.nlp import NLPContext


class LinearCRFLoss(nn.Module):
    def __init__(
        self,
        crf_weight: float = 1,
        cross_entroy_weight: float = 1,
        context: NLPContext | None = None,
        reduction: Literal["sum", "mean"] | None = "mean",
        return_all_loss: bool = False,
    ):
        """Linear CRF(conditional random field) loss

        Args:
            crf_weight (float, optional): weight of crf weight. Defaults to 1.
            cross_entroy_weight (float, optional): weight of cross entropy loss. Defaults to 1.
            context (NlpContext, optional): NLP context. Defaults to None.
            reduction (Literal["sum", "mean"] | None, optional): redution mode. Defaults to "mean".
            return_all_loss (bool, optional): return all loss item. Defaults to False.
        """
        super().__init__()
        if context is None:
            context = NLPContext()

        self.crf_weight = crf_weight
        self.cross_entroy_weight = cross_entroy_weight
        self.context = context
        self.reduction = reduction
        self.return_all_loss = return_all_loss

    def calc_hypotheses_score(
        self,
        x: torch.Tensor,
        transition: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """predict normalization constant over all paths

        Args:
            x (torch.Tensor): predicted probability, shape is (batch_size, sequence_length, output_size)
            transition (torch.Tensor): transition matrix, shape is (output_size, output_size)
            mask (torch.Tensor | None, optional): mask for padding index, shape is (batch_size, sequence_length). Defaults to None.

        Returns:
            torch.Tensor: score, shape is (batch_size, output_size, output_size)
        """
        seq_len = x.size(1)

        # edge potential
        # 1, O, O
        transition_score = transition[None, :, :]

        # P(to_t) * P(to_t|from_t-1) * P(from_t-1)
        # P(from|to) = P(to) * P(to|from) / sum_over_from P(to) * P(to|from)

        # this is "from" state
        # unnormalized log P(from)
        # B, O, 1
        alpha = x[:, 0, :, None]
        # early stopping
        for t in range(1, seq_len):
            # this is "to" state, a.k.a. node potential
            # unnormalized log P(to)
            # B, 1, O
            emission_score = x[:, t, None, :]

            # message passing
            # mask transition to <pad>
            # unnormalized log P(to) + unnormalized log P(to | from)
            # B, O, O
            posterior = emission_score + transition_score

            # marginal, unnormalized log P(to)
            # sum over "from" state, then transform back to log prob
            # B, O, O => B, O => B, O, 1
            alpha = torch.where(
                mask[:, t : t + 1, None].bool(),
                alpha,
                (posterior + alpha).logsumexp(1).unsqueeze(-1),
            )

        # B
        # sum over "to" state, this is path likelihood
        # marginal, unnormalized log P(to)
        return alpha.logsumexp(1).squeeze(-1)

    def calc_null_hypothesis_score(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        transition: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """true path score

        Args:
            x (torch.Tensor): predicted probability, shape is (batch_size, sequence_length, output_size)
            y (torch.Tensor): true token, shape is (batch_size, sequence_length)
            transition (torch.Tensor): transition matrix, shape is (output_size, output_size)
            mask (torch.Tensor | None, optional): mask for padding index, shape is (batch_size, sequence_length). Defaults to None.

        Returns:
            torch.Tensor: score, shape is (batch_size)
        """
        # B, S
        emission_score = x.gather(2, y.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            emission_score *= 1 - mask

        # B, S-1
        transition_score = transition[y[:, :-1], y[:, 1:]]
        if mask is not None:
            transition_score *= 1 - mask[:, 1:]

        # B + B => B
        return transition_score.sum(1) + emission_score.sum(1)

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        transition: torch.Tensor,
    ) -> torch.Tensor:
        """Likelihood Ratio for hypothesis token and real token

        Args:
            pred (torch.Tensor): prediction probability, shape is (batch_size, sequence_length, label_size)
            gt (torch.Tensor): label token, shape is (batch_size, sequence_length)
            transition (torch.Tensor): transition matrix of CRF, shape is (label_size, label_size)

        Returns:
            torch.Tensor: likelihood ratio, a scalar
        """
        batch_size, seq_len = gt.shape

        # B, S
        mask = gt.eq(self.context.padding_idx).int()
        max_length = seq_len - mask.sum(1).min().item()
        pred = pred[:, :max_length]
        gt = gt[:, :max_length]
        mask = mask[:, :max_length]

        if self.cross_entroy_weight > 0:
            ce_loss = F.cross_entropy(
                pred.transpose(-1, -2),
                gt,
                reduction="none" if self.reduction is None else self.reduction,
                ignore_index=self.context.padding_idx,
            )
        else:
            shape = (batch_size, seq_len) if self.reduction is None else (1,)
            ce_loss = torch.zeros(*shape, device=self.context.device)
        if self.reduction is None:
            ce_loss = ce_loss.sum(1)

        # B, S, O
        # node potential
        # pred = pred.log_softmax(-1)

        # O, O
        # transition = transition.log_softmax(-1)

        # B
        forward_score = self.calc_hypotheses_score(pred, transition, mask)
        # B
        gold_score = self.calc_null_hypothesis_score(pred, gt, transition, mask)

        # negative log likelihood ratio
        crf_loss = -(gold_score - forward_score)
        if self.reduction == "sum":
            crf_loss = crf_loss.sum()
        elif self.reduction == "mean":
            crf_loss = crf_loss.mean()

        loss = self.crf_weight * crf_loss + self.cross_entroy_weight * ce_loss

        if self.return_all_loss:
            return loss, crf_loss, ce_loss

        return loss
