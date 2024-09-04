import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torchlake.common.schemas.nlp import NlpContext
from torchlake.common.utils.sequence import pack_sequence, unpack_sequence


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
        is_sequence: bool = False,
        model_class: nn.Module | None = None,
    ):
        """Wrapper for sequence model classifier

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            hidden_dim (int): dimension of hidden layer
            output_size (int, optional): number of features of output. Defaults to 1.
            num_layers (int, optional): number of layers. Defaults to 1.
            bidirectional (bool, optional): is bidirectional layer. Defaults to False.
            context (NlpContext, optional): nlp context. Defaults to NlpContext().
            is_sequence (bool, optional): is output tensor a sequence. Defaults to False.
            model_class (nn.Module | None, optional): nn.Module class as sequence modeling layer. Defaults to None.
        """
        super(SequenceModelWrapper, self).__init__()
        assert issubclass(model_class, nn.Module), "model class is not a nn.module"

        self.factor = 2 if bidirectional else 1
        self.context = context
        self.is_sequence = is_sequence

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

        y = pack_sequence(
            y,
            x.ne(self.embed.padding_idx).sum(dim=1).long().detach().cpu(),
            self.context.padding_idx,
        )

        # ot, (ht, ct)
        # ot: batch_size, seq_len, bidirectional * num_layers * hidden_dim
        # ht: bidirectional * num_layers, batch_size, hidden_dim

        states = (
            hidden_state if len(hidden_state) and hidden_state[0] is not None else None
        )

        ot, states = self.rnn(y, states)

        ot = unpack_sequence(ot, self.context.max_seq_len)

        return ot, states

    def classify(
        self,
        ot: torch.Tensor | None = None,
        ht: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert (
            ot is not None or ht is not None
        ), " Must provide either output or hidden state"

        if self.is_sequence:
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


class SequenceModelFullFeatureExtractor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        context: NlpContext = NlpContext(),
        model_class: nn.Module | None = None,
    ):
        """Full timestep and full layers feature extractor for sequence model

        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            hidden_dim (int): dimension of hidden layer
            num_layers (int, optional): number of layers. Defaults to 1.
            bidirectional (bool, optional): is bidirectional layer. Defaults to False.
            context (NlpContext, optional): nlp context. Defaults to NlpContext().
            model_class (nn.Module | None, optional): nn.Module class as sequence modeling layer. Defaults to None.
        """
        super(SequenceModelFullFeatureExtractor, self).__init__()
        assert issubclass(model_class, nn.Module), "model class is not a nn.module"

        self.factor = 2 if bidirectional else 1
        self.context = context
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=context.padding_idx,
        )
        self.rnns = nn.ModuleList(
            [
                model_class(
                    embed_dim,
                    hidden_dim,
                    num_layers=1,
                    bidirectional=bidirectional,
                    batch_first=True,
                ),
                *[
                    model_class(
                        hidden_dim,
                        hidden_dim,
                        num_layers=1,
                        bidirectional=bidirectional,
                        batch_first=True,
                    )
                    for _ in range(num_layers - 1)
                ],
            ]
        )

    def _rnn_forward(
        self,
        x: torch.Tensor | PackedSequence,
        states: tuple[torch.Tensor] | None = None,
        seq_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        features = []
        states_placeholder = []

        ot = x
        for i, layer in enumerate(self.rnns):
            ot, states = layer(ot, states)
            features.append(ot)
            if self.factor == 2:
                ot = unpack_sequence(ot, self.context.max_seq_len)[
                    :, :, self.hidden_dim :
                ]
                ot = pack_sequence(ot, seq_lengths, self.context.padding_idx)

            # collect hidden states between cells
            is_multiple_state = isinstance(states, tuple)
            if not is_multiple_state:
                states_placeholder.append(states)
            # for RNN variant has more than one hidden state
            else:
                # to init
                if i == 0:
                    for _ in range(len(states)):
                        states_placeholder.append([])
                for placeholder, state in zip(states_placeholder, states):
                    placeholder.append(state)

        # collect output state
        features = [
            unpack_sequence(feature, self.context.max_seq_len) for feature in features
        ]

        # collect hidden state
        if not is_multiple_state:
            states_placeholder = torch.cat(states_placeholder, 0)
        else:
            # XXX: placeholder directly assign not work @@
            for i, placeholder in enumerate(states_placeholder):
                states_placeholder[i] = torch.cat(placeholder, 0)
            states_placeholder = tuple(states_placeholder)

        return torch.cat(features, -1), states_placeholder

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: tuple[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_lengths = x.ne(self.embed.padding_idx).sum(dim=1).long().detach().cpu()
        # batch_size, seq_len, embed_dim
        y = self.embed(x)

        y = pack_sequence(y, seq_lengths, self.context.padding_idx)

        # ot, (ht, ct)
        # ot: batch_size, seq_len, bidirectional * num_layers * hidden_dim
        # ht: bidirectional * num_layers, batch_size, hidden_dim

        ot, hidden_state = self._rnn_forward(y, hidden_state, seq_lengths)

        ot = unpack_sequence(ot, self.context.max_seq_len)

        return ot, hidden_state
