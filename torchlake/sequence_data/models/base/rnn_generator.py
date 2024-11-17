from functools import partial

import torch
from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NlpContext
from torchlake.common.utils.sequence import get_input_sequence

from .rnn_discriminator import RNNDiscriminator


class RNNGenerator(ModelBase):
    def __init__(self, model: RNNDiscriminator, attention=None):
        """Wrapper for sequence model generator

        Args:
            model (RNNDiscriminator): RNN discriminator
            attention (nn.Module, optional): attention module. Defaults to None.
        """
        assert issubclass(
            type(model),
            RNNDiscriminator,
        ), "model is not a RNNDiscriminator"

        super().__init__(
            None,
            None,
            neck_kwargs={"attention": attention},
            head_kwargs={"model": model},
        )

    def build_foot(self, _, **kwargs):
        self.foot = ...

    def build_neck(self, **kwargs):
        self.neck = kwargs.pop("attention")

    def build_head(self, _, **kwargs):
        self.head: RNNDiscriminator = kwargs.pop("model")
        self.head.forward = partial(self.head.forward, output_state=True)

    def train(self, mode=True):
        result = super().train(mode)
        result.head.sequence_output = False
        result.forward = self.loss_forward
        return result

    def eval(self):
        result = super().eval()
        result.head.sequence_output = True
        result.forward = self.predict
        return result

    def attend(
        self,
        ht: torch.Tensor,
        os: torch.Tensor,
        output_score: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """attend hidden state of source hidden state

        Args:
            ht (torch.Tensor): current hidden state
            os (torch.Tensor, optional): source output state. Defaults to None.
            output_score (bool, optional): output attention score or not. Defaults to False.

        Returns:
            torch.Tensor: attended representation
        """
        if self.neck is not None:
            # num_layers = self.head.num_layers
            # if num_layers == 1:
            # ht, score = self.neck(hs, ht)
            # elif num_layers > 1:
            ht, score = self.neck(os, ht)

            if output_score:
                return ht, score

        return ht

    def loss_forward(
        self,
        y: torch.Tensor,
        ht: torch.Tensor | None = None,
        *states: tuple[torch.Tensor],
        ot: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
        output_score: bool = False,
    ) -> torch.Tensor:
        """training loss forward

        Args:
            y (torch.Tensor): groundtruth sequence, shape is (batch_size, seq_len)
            ht (torch.Tensor, optional): hidden state. shape is (bidirectional * num_layers, batch_size, hidden_dim). Defaults to None.
            *states (tuple[torch.Tensor]): other hidden states.
            ot (torch.Tensor, optional): output state. shape is (batch_size, seq_len, bidirectional * hidden_dim). Defaults to None.
            teacher_forcing_ratio (float, optional): scheduled sampling in paper [1506.03099]. Defaults to 0.5.
            output_score (bool, optional): output attention score or not. Defaults to False.

        Returns:
            torch.Tensor: generated sequence
        """
        context: NlpContext = self.head.context
        batch_size = y.size(0)
        output_size = self.head.output_size
        max_seq_len = context.max_seq_len
        device = context.device

        # keep attention source states
        source_states = (ot,)
        # keep attention score
        scores = []

        # tensor to store generated logit
        outputs = [torch.full((batch_size, output_size), -1e4).to(device)]
        # start from <bos>
        outputs[0][:, context.bos_idx] = 1e4

        # next token prediction
        input_seq = get_input_sequence((batch_size, 1), context)
        for t in range(1, max_seq_len, 1):
            # attention
            att = self.attend(ht, *source_states, output_score=output_score)
            # store attention score
            if output_score:
                ht, score = att
                scores.append(score)
            else:
                ht = att

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            # B, V
            output, states = self.head(input_seq, ht, *states)
            ht, states = states[0], states[1:]

            # place predictions in a tensor holding predictions for each token
            # B, V
            outputs.append(output)

            # 1506.03099: teacher forcing
            # decide if we are going to use teacher forcing or not
            # use actual next token as next input
            # if not, use the highest predicted token
            # B, 1
            input_seq = torch.where(
                torch.rand(batch_size, 1).lt(teacher_forcing_ratio).to(device),
                y[:, t : t + 1],
                output.argmax(-1, keepdim=True),
            )

            # early stopping
            if (
                input_seq[:, -1].eq(context.eos_idx).all()
                or input_seq[:, -1].eq(context.padding_idx).all()
            ):
                break

        # B, ?, V
        outputs = torch.stack(outputs, 1)
        # pad it back
        need_pad_length = max_seq_len - outputs.size(1)
        if need_pad_length > 0:
            outputs = torch.cat(
                [
                    outputs,
                    torch.full((batch_size, need_pad_length, output_size), -1e4).to(
                        device
                    ),
                ],
                1,
            )
            outputs[:, -need_pad_length:, context.padding_idx] = 1e4

        if output_score:
            # B, S, V # D, B, S, S'
            return outputs, torch.stack(scores, -1)
        else:
            # B, S, V
            return outputs

    def predict(
        self,
        primer: torch.Tensor | None = None,
        ht: torch.Tensor | None = None,
        *states: tuple[torch.Tensor],
        ot: torch.Tensor | None = None,
        topk: int = 1,
        output_score: bool = False,
    ) -> torch.Tensor:
        """predict sequence with beam search

        Args:
            primer (torch.Tensor|None, optional): leading sequence, shape in (batch, seq). Defaults to None.
            ht (torch.Tensor, optional): hidden state. shape is (bidirectional * num_layers, batch_size, hidden_dim). Defaults to None.
            *states (tuple[torch.Tensor]): other hidden states.
            ot (torch.Tensor, optional): output state. shape is (batch_size, seq_len, bidirectional * hidden_dim). Defaults to None.
            topk (int, optional): beam search size. Defaults to 1.
            output_score (bool, optional): output attention score or not. Defaults to False.

        Returns:
            torch.Tensor: output sequence
        """
        assert topk >= 1, "Top k should be at least 1"

        context: NlpContext = self.head.context
        max_seq_len = context.max_seq_len
        device = context.device
        output_size = self.head.output_size

        # keep attention source states
        source_states = (ot,)
        # keep attention score
        scores = []

        # next token prediction (first one)
        if primer is None:
            input_seq = get_input_sequence((1, 1), context)
        else:
            input_seq = primer
        batch_size = input_seq.size(0)

        # (B, 1, V), (D, B, h)
        output, states = self.head(input_seq, ht, *states)
        # D, topk*B, h
        states = tuple(state.repeat(1, topk, 1) for state in states)

        # B, 1, topk
        topk_values, topk_indices = output.topk(topk, dim=-1)
        # beam search has topk hypothses
        # topk, B, 2
        paths = torch.cat(
            [input_seq.unsqueeze_(-1).repeat(1, 1, topk), topk_indices], 1
        ).permute(2, 0, 1)
        # B, topk
        probs: torch.Tensor = topk_values.squeeze_(1)
        # beam search forward
        for t in range(1, max_seq_len, 1):
            # attention
            att = self.attend(ht, *source_states, output_score=output_score)
            # store attention score
            if output_score:
                ht, score = att
                scores.append(score)
            else:
                ht = att

            # topk, B, V
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, states = self.head(
                paths.reshape(-1, paths.size(-1)),
                *states,
            )
            output: torch.Tensor = output.view(topk, batch_size, 1 + t, -1)[:, :, -1]

            # B, topk
            topk_values, topk_indices = (
                output.permute(1, 0, 2).reshape(batch_size, -1).topk(topk, dim=-1)
            )

            # B, topk
            parents = topk_indices // output_size

            # topk, B, t
            paths = torch.cat(
                [
                    paths[parents.T, torch.arange(batch_size)],
                    topk_indices.T.unsqueeze(-1) % output_size,
                ],
                -1,
            )
            probs[
                torch.arange(batch_size).unsqueeze_(-1).to(device),
                parents,
            ] += topk_values

            # early stopping
            if (
                paths[:, :, -1].eq(context.eos_idx).all()
                or paths[:, :, -1].eq(context.padding_idx).all()
            ):
                break

        # B, S
        outputs = paths[probs.argmax(-1), torch.arange(batch_size).to(device)]

        if output_score:
            # B, S, V # D, B, S, S'
            return outputs, torch.stack(scores, -1)
        else:
            return outputs
