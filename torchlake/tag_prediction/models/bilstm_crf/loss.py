import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext


class LinearCRFLoss(nn.Module):
    def __init__(self, context: NlpContext = NlpContext()):
        """Linear CRF(conditional random field) loss

        Args:
            context (NlpContext, optional): nlp context. Defaults to NlpContext().
        """
        super().__init__()
        self.context = context

    def calc_hypotheses_score(
        self,
        x: torch.Tensor,
        transition: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """predict path score

        Args:
            x (torch.Tensor): predicted probability, shape is (batch_size, sequence_length, output_size)
            transition (torch.Tensor): transition matrix, shape is (output_size, output_size)
            mask (torch.Tensor | None, optional): mask for padding index, shape is (batch_size, sequence_length). Defaults to None.

        Returns:
            torch.Tensor: score, shape is (batch_size, output_size, output_size)
        """
        seq_len = x.size(1)

        # B, S, O, 1
        x = x.unsqueeze(-1)

        # edge potential
        # 1, O, O
        transition_score = transition[None, :, :].log_softmax(-1)

        # stop criteria
        non_pad_length = seq_len - mask.sum(1).max().item()

        # forward
        # B, O
        alpha = x[:, 0, :, :].squeeze(-1)
        for t in range(1, seq_len):

            # node potential
            # B, O, 1
            emit_score = x[:, t, :, :]

            # message passing
            # B, O
            alpha = alpha + (emit_score + transition_score).sum(1)

            # early stopping
            if non_pad_length == t:
                break

        return alpha

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
        emit_score = x.gather(2, y.unsqueeze(-1)).squeeze(-1)

        # B, S-1
        transition_score = transition.log_softmax(-1)[y[:, :-1], y[:, 1:]]
        if mask is not None:
            transition_score *= 1 - mask[:, 1:]

        # B, S-1 + B, S-1 + B,1 => B, S-1
        # emit score of `to` token is what we need
        # plus emit score of bos
        score = transition_score + emit_score[:, 1:] + emit_score[:, 0:1]

        # B
        return score.sum(1)

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
        # B, S
        mask = gt.eq(self.context.padding_idx).int()

        # B, S, O
        pred *= 1 - mask[:, :, None]
        pred = pred.log_softmax(-1)

        # B, O
        forward_score = self.calc_hypotheses_score(pred, transition, mask)
        # B
        gold_score = self.calc_null_hypothesis_score(pred, gt, transition, mask)
        return (forward_score - gold_score[:, None]).mean()
