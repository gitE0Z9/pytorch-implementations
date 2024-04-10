import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torchlake.common.schemas.nlp import NlpContext


class SequenceModelWrapper(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        num_layers: int = 1,
        bidirectional: bool = False,
        context: NlpContext = NlpContext(),
        is_token: bool = False,
        model_class: nn.Module | None = None,
    ):
        super(SequenceModelWrapper, self).__init__()
        assert issubclass(model_class, nn.Module), "model class is not a nn.module"

        self.factor = 2 if bidirectional else 1
        self.context = context
        self.is_token = is_token

        self.embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=context.padding_idx,
        )
        self.rnn = model_class(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * self.factor)
        self.fc = nn.Linear(hidden_dim * self.factor, output_size)

    def feature_extract(
        self,
        x: torch.Tensor,
        *hidden_state: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # batch_size, seq_len, embed_dim
        y = self.embed(x)

        if self.embed.padding_idx is not None and y.size(1) > 1:
            y = pack_padded_sequence(
                y,
                x.ne(self.embed.padding_idx).sum(dim=1).long().detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

        # ot, (ht, ct)
        # ot: batch_size, seq_len, bidirectional*hidden_dim
        # ht: bidirectional * layer_size, batch_size, hidden_dim

        states = hidden_state if len(hidden_state) else None

        ot, states = self.rnn(y, states)

        if isinstance(ot, PackedSequence):
            ot, _ = pad_packed_sequence(
                ot,
                batch_first=True,
                total_length=self.context.max_seq_len,
            )

        return ot, states

    def classify(
        self,
        ot: torch.Tensor | None = None,
        ht: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert (
            ot is not None or ht is not None
        ), " Must provide either output or hidden state"

        if self.is_token:
            y = self.layer_norm(ot)
        else:
            # the deepest layer's hidden state recept the whole sequence
            y = torch.cat([ht[-2], ht[-1]], -1) if self.factor == 2 else ht[-1]

        return self.fc(y)

    def forward(
        self,
        x: torch.Tensor,
        *hidden_state: torch.Tensor | None,
        output_state: bool = False,
    ) -> torch.Tensor:
        ot, states = self.feature_extract(x, *hidden_state)

        h = states[0] if isinstance(states, tuple) else states

        if output_state:
            return self.classify(ot, h), states
        else:
            return self.classify(ot, h)
