import torch
from torch import nn

from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NLPContext
from torchlake.common.utils.sequence import pack_sequence, unpack_sequence


class RNNDiscriminator(ModelBase):
    def __init__(
        self,
        input_channel: int,
        hidden_dim: int,
        output_size: int,
        context_dim: int = 0,
        num_layers: int = 1,
        bidirectional: bool = False,
        output_sequence: bool = False,
        drop_fc: bool = False,
        context: NLPContext | None = None,
        rnn_class: nn.Module | None = None,
        projection_layer: nn.Module | None = None,
    ):
        """Wrapper for sequence model discriminator

        Args:
            input_channel (int): input channel size
            hidden_dim (int): dimension of hidden layer
            output_size (int, optional): number of features of output.
            context_dim (int, optional): dimension of context vector. Defaults to 0.
            num_layers (int, optional): number of layers. Defaults to 1.
            bidirectional (bool, optional): is bidirectional layer. Defaults to False.
            output_sequence (bool, optional): every position of a sequence has output. Defaults to False.
            drop_fc (bool, optional): remove fully connected head. Defaults to False.
            context (NlpContext, optional): NLP context. Defaults to None.
            rnn_class (nn.Module | None, optional): nn.Module class as sequence modeling layer. Defaults to None.
            projection_layer (nn.Module | None, optional): nn.Module class as projection layer. Defaults to None.
        """
        assert issubclass(rnn_class, nn.Module), "rnn class is not a nn.module"
        if context is None:
            context = NLPContext()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_sequence = output_sequence
        self.drop_fc = drop_fc
        self.context = context

        self.factor = 2 if bidirectional else 1

        super().__init__(
            input_channel,
            output_size,
            foot_kwargs={
                "projection_layer": projection_layer,
            },
            blocks_kwargs={
                "rnn_class": rnn_class,
            },
        )

    @property
    def feature_dim(self) -> int:
        return self.hidden_dim * self.factor

    def set_embedding_as_projection_layer(self, vocab_size: int, embed_dim: int):
        self.foot = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=self.context.padding_idx,
        )

    def build_foot(self, _: int, **kwargs):
        projection_layer = kwargs.pop("projection_layer")

        if projection_layer is None:
            self.foot = nn.Identity()
        else:
            self.foot = projection_layer

    def build_blocks(self, **kwargs):
        rnn_class = kwargs.pop("rnn_class")

        self.blocks = rnn_class(
            self.input_channel + self.context_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def build_neck(self):
        self.neck = nn.LayerNorm(self.feature_dim)

    def build_head(self, output_size: int):
        if not self.drop_fc:
            self.head = nn.Linear(self.feature_dim, output_size)
        else:
            self.head = nn.Identity()

    def feature_extract(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        ht: torch.Tensor | None = None,
        *states: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor] | None]:
        """extract feature from RNN

        Args:
            x (torch.Tensor): input. shape is (batch_size, seq_len)
            y (torch.Tensor): embedded input. shape is (batch_size, seq_len, embed_dim)
            ht (torch.Tensor, optional): hidden state. shape is (bidirectional * num_layers, batch_size, hidden_dim). Defaults to None.
            *states (tuple[torch.Tensor]): other hidden states.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: output and hidden states
        """
        seq_len = y.size(1)
        if self.context.padding_idx is not None and self.input_channel == 0:
            y = pack_sequence(
                y,
                x.ne(self.context.padding_idx).sum(dim=1).long().detach().cpu(),
                self.context.padding_idx,
            )

        # o, (ht, ct)
        # o: batch_size, seq_len, bidirectional * num_layers * hidden_dim
        # ht: bidirectional * num_layers, batch_size, hidden_dim

        if ht is not None:
            states = (ht, *states)

        # for compatibility with torch
        # e.g. rnn & gru only receive tensor & None
        # e.g. lstm only receive tuple[tensor, tensor] & None
        if len(states) == 0:
            states = None
        elif len(states) == 1:
            states = states[0]

        o, states = self.blocks(y, states)

        if isinstance(states, tuple):
            ht, states = states[0], states[1:]
        else:
            ht, states = states, tuple()

        if self.context.padding_idx is not None:
            o = unpack_sequence(o, seq_len)

        return o, ht, states

    def classify(
        self,
        o: torch.Tensor | None = None,
        ht: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """classify for sequence output or single output

        Args:
            o (torch.Tensor | None, optional): output sequence. Defaults to None.
            ht (torch.Tensor | None, optional): hidden state. Defaults to None.

        Returns:
            torch.Tensor: result
        """
        assert (
            o is not None or ht is not None
        ), " Must provide either output or hidden state"

        if self.output_sequence:
            y = self.neck(o)
        else:
            _, b, _ = ht.shape
            # hidden state of the deepest layer's last timestamp represent the whole sequence
            y = ht[-self.factor :].transpose(0, 1).reshape(b, -1)

        return self.head(y)

    def forward(
        self,
        x: torch.Tensor,
        ht: torch.Tensor | None = None,
        *states: tuple[torch.Tensor],
        context_vector: torch.Tensor | None = None,
        output_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor]]:
        """forward

        Args:
            x (torch.Tensor): input. shape is (batch_size, seq_len) or (batch_size, seq_len, dim)
            ht (torch.Tensor, optional): hidden state. shape is (bidirectional * num_layers, batch_size, hidden_dim). Defaults to None.
            *states (tuple[torch.Tensor]): other hidden states.
            context_vector (torch.Tensor, optional): context vector from attention. shape is (batch_size, 1, context_dim). Defaults to None.
            output_state (bool, optional): also return hidden state. Defaults to False.

        Returns:
            torch.Tensor: prediction or prediction and states
        """
        # enable_embed: batch_size, seq_len, embed_dim
        # otherwise: batch_size, seq_len, dim
        y = self.foot(x)

        if context_vector is not None:
            # batch_size, seq_len, (embed_dim or dim) + context_dim
            y = torch.cat([y, context_vector], -1)

        o, ht, states = self.feature_extract(x, y, *(ht, *states))

        y = self.classify(o, ht)

        if output_state:
            return y, (ht, *states)

        return y
