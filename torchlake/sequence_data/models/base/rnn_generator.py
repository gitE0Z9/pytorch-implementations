from functools import partial

import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext
from torchlake.common.utils.sequence import get_input_sequence

from .rnn_discriminator import RNNDiscriminator

from torch.nn.utils.rnn import pad_sequence


class RNNGenerator(nn.Module):
    def __init__(self, model: RNNDiscriminator):
        """Wrapper for sequence model generator

        Args:
            model (RNNDiscriminator): RNN discriminator
        """
        super().__init__()
        assert isinstance(model, RNNDiscriminator), "model is not a RNNDiscriminator"

        self.model = model
        self.model.forward = partial(self.model.forward, output_state=True)

    def train(self, mode=True):
        result = super().train(mode)
        result.model.sequence_output = False
        result.forward = self.loss_forward
        return result

    def eval(self):
        result = super().eval()
        result.model.sequence_output = True
        result.forward = self.predict
        return result

    def loss_forward(
        self,
        y: torch.Tensor,
        ht: torch.Tensor | None = None,
        *states: tuple[torch.Tensor],
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """training loss forward

        Args:
            y (torch.Tensor): groundtruth sequence, shape is (batch_size, seq_len)
            ht (torch.Tensor, optional): hidden state. shape is (bidirectional * num_layers, batch_size, hidden_dim). Defaults to None.
            *states (tuple[torch.Tensor]): other hidden states.
            teacher_forcing_ratio (float, optional): scheduled sampling in paper [1506.03099]. Defaults to 0.5.

        Returns:
            torch.Tensor: generated sequence
        """
        context: NlpContext = self.model.context
        batch_size = y.size(0)
        output_size = self.model.output_size
        max_seq_len = context.max_seq_len
        device = context.device

        # tensor to store generated logit
        outputs = [torch.full((batch_size, output_size), -1e4).to(device)]
        # start from <bos>
        outputs[0][:, context.bos_idx] = 1e4

        # next token prediction (first one)
        input_seq = get_input_sequence((batch_size, 1), context)
        for t in range(1, max_seq_len, 1):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            # B, V
            output, states = self.model.forward(input_seq, ht, *states)
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
                    torch.full(
                        (batch_size, need_pad_length, self.model.output_size), -1e4
                    ).to(device),
                ],
                1,
            )
            outputs[:, -need_pad_length:, context.padding_idx] = 1e4

        # B, S, V
        return outputs

    def predict(
        self,
        primer: torch.Tensor | None = None,
        ht: torch.Tensor | None = None,
        *states: tuple[torch.Tensor],
        topk: int = 1,
    ) -> torch.Tensor:
        """predict sequence with beam search

        Args:
            primer (torch.Tensor|None, optional): leading sequence, shape in (batch, seq). Defaults to None.
            ht (torch.Tensor, optional): hidden state. shape is (bidirectional * num_layers, batch_size, hidden_dim). Defaults to None.
            *states (tuple[torch.Tensor]): other hidden states.
            topk (int, optional): beam search size. Defaults to 1.

        Returns:
            torch.Tensor: output sequence
        """
        assert topk >= 1, "Top k should be at least 1"

        context: NlpContext = self.model.context
        max_seq_len = context.max_seq_len
        device = context.device
        output_size = self.model.output_size

        # next token prediction (first one)
        if primer is None:
            input_seq = get_input_sequence((1, 1), context)
        else:
            input_seq = primer
        batch_size = input_seq.size(0)

        # (B, 1, V), (D, B, h)
        output, states = self.model.forward(input_seq, ht, *states)
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
            # topk, B, V
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, states = self.model.forward(
                paths.reshape(-1, paths.size(-1)),
                *states,
            )
            output = output.view(topk, batch_size, 1 + t, -1)[:, :, -1]

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
        return paths[probs.argmax(-1), torch.arange(batch_size).to(device)]
