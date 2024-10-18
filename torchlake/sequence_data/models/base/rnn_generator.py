from functools import partial

import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext

from .rnn_discriminator import RNNDiscriminator


class RNNGenerator(nn.Module):
    def __init__(
        self,
        input_channel: int,
        model: RNNDiscriminator,
    ):
        """Wrapper for sequence model generator

        Args:
            input_channel (int): input channel size
            model (RNNDiscriminator): RNN discriminator
        """
        super().__init__()
        assert isinstance(model, RNNDiscriminator), "model is not a RNNDiscriminator"

        self.input_channel = input_channel
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
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """training loss forward

        Args:
            y (torch.Tensor): groundtruth sequence, shape is (batch_size, seq_len)
            teacher_forcing_ratio (float, optional): scheduled sampling in paper [1506.03099]. Defaults to 0.5.

        Returns:
            torch.Tensor: generated sequence
        """
        context: NlpContext = self.model.context
        batch_size = y.size(0)
        vocab_size = self.input_channel
        max_seq_len = context.max_seq_len
        device = context.device

        # tensor to store generated log likelihood
        outputs = [torch.full((batch_size, vocab_size), 1e-4).to(device)]
        # start from <bos>
        outputs[0][:, context.bos_idx] = 0

        internal_states = tuple()

        # next token prediction (first one)
        input_seq = torch.full((batch_size, 1), context.bos_idx).to(device)
        for t in range(1, max_seq_len, 1):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            # B, h
            output, internal_states = self.model.forward(input_seq, *internal_states)

            # place predictions in a tensor holding predictions for each token
            # B, h
            outputs.append(output)

            # 1506.03099: teacher forcing
            # decide if we are going to use teacher forcing or not
            # use actual next token as next input
            # if not, use the highest predicted token
            # B, 1+t
            input_seq = torch.cat(
                [
                    input_seq,
                    torch.where(
                        torch.rand(batch_size, 1).lt(teacher_forcing_ratio).to(device),
                        y[:, t : t + 1],
                        output.argmax(-1, keepdim=True),
                    ),
                ],
                -1,
            )

        # B, S, V
        outputs = torch.stack(outputs, 1)

        return outputs

    def predict(self, x: torch.Tensor, topk: int = 1) -> torch.Tensor:
        """predict sequence with beam search

        Args:
            x (torch.Tensor): source sequence, shape in (batch, seq)
            topk (int, optional): beam search size. Defaults to 1.

        Returns:
            torch.Tensor: output sequence
        """
        assert topk >= 1, "Top k should be at least 1"

        context: NlpContext = self.model.context
        batch_size = x.size(0)
        max_seq_len = context.max_seq_len
        device = context.device

        # next token prediction (first one)
        input_seq = torch.full((batch_size, 1), context.bos_idx).to(device)
        # B, 1, V, (D, B, h)
        output, internal_states = self.model(input_seq)
        internal_states = tuple(state.repeat(1, topk, 1) for state in internal_states)
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
            output, internal_states = self.model.forward(
                paths.reshape(-1, paths.size(-1)),
                *internal_states,
            )
            output = output.view(topk, batch_size, 1 + t, -1)[:, :, -1]

            # B, topk
            topk_values, topk_indices = (
                output.permute(1, 0, 2).reshape(batch_size, -1).topk(topk, dim=-1)
            )

            # B, topk
            parents = topk_indices // self.input_channel

            # topk, B, t
            paths = torch.cat(
                [
                    paths[parents.T, torch.arange(batch_size)],
                    topk_indices.T.unsqueeze(-1) % self.input_channel,
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
