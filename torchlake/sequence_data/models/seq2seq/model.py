import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext
from torchlake.sequence_data.utils.decode import beam_search

from .network import (
    Seq2SeqDecoder,
    Seq2SeqEncoder,
    Seq2SeqAttentionEncoder,
    GlobalAttention,
    LocalAttention,
)


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Seq2SeqEncoder | Seq2SeqAttentionEncoder,
        decoder: Seq2SeqDecoder,
        attention: GlobalAttention | LocalAttention | None = None,
        context: NlpContext = NlpContext(),
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention

        self.context = context

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """last hidden state of the encoder is used as the initial hidden state of the decoder

        Args:
            x (torch.Tensor): source sentence

        Returns:
            torch.Tensor: encoding
        """
        # 1409.3215 p.3: reverse x
        return self.encoder(x.flip(1))

    def attend_hidden_state(
        self,
        hs: torch.Tensor,
        internal_state: torch.Tensor | tuple[torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """_summary_

        Args:
            hs (torch.Tensor): _description_
            internal_state (torch.Tensor | tuple[torch.Tensor]): _description_

        Returns:
            torch.Tensor: _description_
        """
        if self.attention is not None:
            is_complex_state = isinstance(internal_state, tuple)
            ht = internal_state[0] if is_complex_state else internal_state
            ht, _ = self.attention(hs, ht)
            if is_complex_state:
                internal_state = (ht, *internal_state[1:])
            else:
                internal_state = ht

        return internal_state

    def loss_forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        seq_len = y.size(1)
        y_vocab_size = self.decoder.rnn.embed.num_embeddings

        # decide if we are going to use teacher forcing or not
        enable_teacher_force = (
            torch.rand((batch_size, seq_len - 1, 1))
            .lt(teacher_forcing_ratio)
            .to(self.context.device)
        )

        # encoding
        hs, internal_state = self.encode(x)

        # tensor to store decoder outputs
        outputs = torch.full((batch_size, seq_len, y_vocab_size), -1e-4).to(
            self.context.device
        )
        outputs[:, 0, self.context.bos_idx] = 0
        # next token prediction
        input_seq = torch.full((batch_size, 1), self.context.bos_idx).to(
            self.context.device
        )
        for t in range(1, seq_len, 1):
            #  attention
            internal_state = self.attend_hidden_state(hs, internal_state)

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, internal_state = self.decoder(input_seq, *internal_state)

            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = output

            # 1506.03099: teacher forcing
            # use actual next token as next input
            # if not, use the highest predicted token
            input_seq = torch.where(
                enable_teacher_force[:, t - 1],
                y[:, t : t + 1],
                output.argmax(-1, keepdim=True),
            )

        return outputs

    def predict(self, x: torch.Tensor, topk: int = 1) -> torch.Tensor:
        """predict sequence with beam search

        Args:
            x (torch.Tensor): source sequence, shape in (seq,)
            topk (int, optional): beam search size. Defaults to 1.

        Returns:
            torch.Tensor: output sequence
        """
        if topk < 1:
            raise NotImplementedError("Top k should be at least 1")

        is_beam_search = topk > 1
        is_greedy = topk > 1

        batch_size = x.size(0)
        seq_len = self.context.max_seq_len

        # encoding
        hs, internal_state = self.encode(x)

        # beam search forward
        hypotheses = []
        input_seq = torch.full((batch_size, 1), self.context.bos_idx).to(
            self.context.device
        )

        if is_greedy:
            hypotheses.append(input_seq)

        for _ in range(1, seq_len, 1):
            # attention
            internal_state = self.attend_hidden_state(hs, internal_state)

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, internal_state = self.decoder(input_seq, *internal_state)

            # B, 1, topk
            if is_beam_search:
                topk_values, topk_indices = output.topk(topk, dim=-1)
                hypotheses.append((topk_values, topk_indices))

                # topk * B, 1
                input_seq = (topk_indices % topk).permute(2, 0, 1).reshape(-1, seq_len)
                # TODO: internal state repeat k times
            elif is_greedy:
                # B, 1
                input_seq = output.argmax(-1)
                hypotheses.append(input_seq)

        if is_beam_search:
            # beam search backward
            # B, S
            return beam_search(
                hypotheses,
                batch_size,
                topk,
                self.context,
            )
        elif is_greedy:
            # B, S
            return torch.cat(hypotheses, -1)
