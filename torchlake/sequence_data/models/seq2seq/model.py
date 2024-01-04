import random

import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext

from .network import Seq2SeqDecoder, Seq2SeqEncoder


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        decoder: Seq2SeqDecoder,
        context: NlpContext = NlpContext(),
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.context = context

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size, y_seq_len = y.shape
        y_vocab_size = self.decoder.rnn.fc.out_features

        # tensor to store decoder outputs
        outputs = torch.full(
            (batch_size, y_seq_len, y_vocab_size),
            float(self.context.bos_idx),
        ).to(self.context.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        internal_state = self.encoder(x)

        # first input to the decoder is the <sos> tokens
        # input_seq = y[:, 0:1]

        # for t in range(1, y_seq_len):
        #     # insert input token embedding, previous hidden and previous cell states
        #     # receive output tensor (predictions) and new hidden and cell states
        output = self.decoder(y, *internal_state)

        #     # place predictions in a tensor holding predictions for each token
        #     outputs[:, t] = output

        #     # decide if we are going to use teacher forcing or not
        #     enable_teacher_force = random.random() < teacher_forcing_ratio

        #     # get the highest predicted token from our predictions
        #     top1 = output.argmax(1)

        #     # if teacher forcing, use actual next token as next input
        #     # if not, use predicted token
        #     input_seq = y[:, t : t + 1] if enable_teacher_force else top1

        return output
