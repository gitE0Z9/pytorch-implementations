import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext

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

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        topk: int = 1,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        hs, internal_state = self.encode(x)

        batch_size = x.size(0)
        y_seq_len = y.size(1)
        y_vocab_size = self.decoder.rnn.embed.num_embeddings

        # tensor to store decoder outputs
        outputs = torch.full(
            (batch_size, y_seq_len, y_vocab_size),
            float(self.context.bos_idx),
        ).to(self.context.device)

        # decide if we are going to use teacher forcing or not
        enable_teacher_force = (
            torch.rand((batch_size, y_seq_len, 1))
            .lt(teacher_forcing_ratio)
            .to(self.context.device)
        )

        #  attention
        internal_state = self.attend_hidden_state(hs, internal_state)

        # next token prediction
        input_seq = outputs[:, 0:1, 0]
        for t in range(y_seq_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, internal_state = self.decoder(input_seq.int(), *internal_state)
            internal_state = self.attend_hidden_state(hs, internal_state)

            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = output

            # get the highest predicted token from our predictions
            top1 = output.argmax(-1, keepdim=True)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            proposed_next_token = torch.where(
                enable_teacher_force[:, t], y[:, t + 1 : t + 2], top1
            )
            input_seq = proposed_next_token

        return outputs
