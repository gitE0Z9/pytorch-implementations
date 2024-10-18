import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext
from torchlake.common.utils.sequence import pack_sequence, unpack_sequence

from torchlake.common.models.model_base import ModelBase


class RNNDiscriminator(ModelBase):
    def __init__(
        self,
        input_channel: int,
        embed_dim: int = 300,
        hidden_dim: int = 128,
        output_size: int = 1,
        num_layers: int = 1,
        bidirectional: bool = False,
        sequence_output: bool = False,
        enable_embed: bool = True,
        drop_fc: bool = False,
        context: NlpContext = NlpContext(),
        model_class: nn.Module | None = None,
    ):
        """Wrapper for sequence model discriminator

        Args:
            input_channel (int): input channel size
            embed_dim (int, optional): dimension of embedding vector. Defaults to 300.
            hidden_dim (int): dimension of hidden layer
            output_size (int, optional): number of features of output. Defaults to 1.
            num_layers (int, optional): number of layers. Defaults to 1.
            bidirectional (bool, optional): is bidirectional layer. Defaults to False.
            sequence_output (bool, optional): is output tensor a sequence. Defaults to False.
            enable_embed (bool, optional): need an embedding layer. Defaults to True.
            drop_fc (bool, optional): remove fully connected head. Defaults to False.
            context (NlpContext, optional): nlp context. Defaults to NlpContext().
            model_class (nn.Module | None, optional): nn.Module class as sequence modeling layer. Defaults to None.
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sequence_output = sequence_output
        self.enable_embed = enable_embed
        self.drop_fc = drop_fc
        self.context = context
        self.model_class = model_class

        self.factor = 2 if bidirectional else 1

        super().__init__(input_channel, output_size)
        assert issubclass(model_class, nn.Module), "model class is not a nn.module"

    @property
    def feature_dim(self) -> int:
        return self.hidden_dim * self.factor

    def build_foot(self, input_channel: int):
        if self.enable_embed:
            self.foot = nn.Embedding(
                input_channel,
                self.embed_dim,
                padding_idx=self.context.padding_idx,
            )
        else:
            self.foot = nn.Identity()

    def build_blocks(self):
        self.blocks = self.model_class(
            self.embed_dim,
            self.hidden_dim,
            self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
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
            y (torch.Tensor): input. shape is (batch_size, seq_len, embed_dim)
            ht (torch.Tensor, optional): hidden state. shape is (bidirectional * num_layers, batch_size, hidden_dim). Defaults to None.
            *states (tuple[torch.Tensor]): other hidden states.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: output and hidden states
        """
        _, seq_len, _ = y.shape
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

        states = states if len(states) and states[0] is not None else None

        o, states = self.blocks(y, states)

        if isinstance(states, tuple):
            ht, states = states[0], states[1:]
        else:
            ht, states = states, tuple()

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

        if self.sequence_output:
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
        output_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor]]:
        """forward

        Args:
            x (torch.Tensor): input. shape is (batch_size, seq_len)
            ht (torch.Tensor, optional): hidden state. shape is (bidirectional * num_layers, batch_size, hidden_dim). Defaults to None.
            *states (tuple[torch.Tensor]): other hidden states.
            output_state (bool, optional): also return hidden state. Defaults to False.

        Returns:
            torch.Tensor: prediction or prediction and states
        """
        # batch_size, seq_len, embed_dim
        y = self.foot(x)

        o, ht, states = self.feature_extract(x, y, *(ht, *states))

        y = self.classify(o, ht)

        if output_state:
            return y, (ht, *states)

        return y
