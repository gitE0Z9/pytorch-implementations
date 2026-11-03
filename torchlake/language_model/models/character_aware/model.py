import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from torchlake.common.models import HighwayBlock
from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NLPContext
from torchlake.sequence_data.models.lstm import LSTMDiscriminator

from .network import CharacterCNNEmbedding


class CharaceterAwareLM(ModelBase):

    def __init__(
        self,
        char_size: int,
        vocab_size: int,
        embed_dim: int,
        hidden_dims: list[int],
        kernels: list[int],
        recurrent_hidden_dim: int,
        num_highway_layers: int = 1,
        num_recurrent_layers: int = 2,
        max_char_size: int = 65,
        context: NLPContext | None = None,
    ):
        """Characeter Aware Language Model in paper[1508.06615v4]

        Args:
            char_size (int): size of characters
            vocab_size (int): size of vocabulary
            embed_dim (int): dimension of embedding vector
            hidden_dim (list[int]): dimension of convolution layers.
            kernels (list[int], optional): size of kernels.
            recurrent_hidden_dim (int): dimension of hidden state of lstm.
            num_highway_layers (int, optional): number of highway layers. Defaults to 1.
            num_recurrent_layers (int, optional): number of recurrent layers. Defaults to 2.
            max_char_size (int, optional): maximum character size of a word.Defaults to 65.
            context (NLPContext, optional): NLP context. Defaults to None.
        """
        if context is None:
            context = NLPContext()

        self.char_size = char_size
        self.embed_dim = embed_dim
        self.context = context
        self.hidden_dims = hidden_dims
        self.kernels = kernels
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.num_recurrent_layers = num_recurrent_layers
        self.num_highway_layers = num_highway_layers
        self.max_char_size = max_char_size
        super().__init__(char_size, vocab_size)

    @property
    def input_embedding(self):
        raise NotImplementedError("please use self.get_word_vectors")

    @property
    def output_embedding(self) -> nn.Embedding:
        w = self.head.head.weight.detach()
        out_c, in_c = w.shape
        e = nn.Embedding(
            out_c,
            in_c,
            padding_idx=self.head.context.padding_idx,
        )
        e.weight.data.copy_(w)

        return e

    @property
    def feature_dim(self) -> int:
        return self.num_recurrent_layers * self.recurrent_hidden_dim

    def build_foot(self, input_channel, **kwargs):
        self.foot = CharacterCNNEmbedding(
            input_channel,
            self.embed_dim,
            hidden_dims=self.hidden_dims,
            kernels=self.kernels,
            max_char_size=self.max_char_size,
            context=self.context,
        )

    def build_neck(self, **kwargs):
        h = sum(self.hidden_dims)
        self.neck = nn.Sequential(
            *[
                HighwayBlock(
                    nn.Sequential(
                        nn.Linear(h, h),
                        nn.ReLU(True),
                    ),
                    nn.Linear(h, h),
                )
                for _ in range(self.num_highway_layers)
            ],
        )

    def build_head(self, output_size, **kwargs):
        self.head = LSTMDiscriminator(
            sum(self.hidden_dims),
            self.recurrent_hidden_dim,
            output_size,
            output_sequence=True,
            num_layers=self.num_recurrent_layers,
            context=self.context,
        )

    def get_word_vectors(
        self,
        ngrams: list[torch.Tensor],
        word_spans: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """forward

        Args:
            ngrams (list[torch.Tensor]): ngram tokens, shape is batch_size*neighbor_size x (#grams)
            word_spans (list[torch.Tensor]): word lengths, shape is batch_size*neighbor_size x (#subsequence)

        Returns:
            list[torch.Tensor]: word vectors after highway layers, shape is batch size x (how many word in a sentence, sum of hidden dim)
        """
        # how many word in all sentence, sum of hidden dims
        y = self.foot(ngrams, word_spans)
        # how many word in all sentence, sum of hidden dims
        y = self.neck(y)
        # batch size x (how many word in a sentence, sum of hidden dims)
        y = y.split([len(word_span) for word_span in word_spans])

        return y

    def forward(
        self,
        ngrams: list[torch.Tensor],
        word_spans: list[torch.Tensor],
    ) -> torch.Tensor:
        """forward

        Args:
            ngrams (list[torch.Tensor]): ngram tokens, length is batch size, the shape of each tensor is #ngram in a sequence
            word_spans (list[torch.Tensor]): word lengths, length is batch size, the shape of each tensor is seq_len

        Returns:
            torch.Tensor: probability of next word depend on all previous words. P(y_t | y_1 ... y_t-1), shape is batch_size, max_seq_len - 1, vocab_size
        """
        # batch size x (how many word in a sentence, sum of hidden dims)
        y = self.get_word_vectors(ngrams, word_spans)
        # batch size, max seq_len in batch, hidden_dim
        y = pad_sequence(y, batch_first=True)
        # batch_size, max seq_len in batch, vocab_size
        y = self.head(y)

        return F.pad(y, (0, 0, 0, self.context.max_seq_len - 1 - y.size(1)))


def characeter_aware_lm_small(
    char_size: int, vocab_size: int, context: NLPContext
) -> CharaceterAwareLM:
    """constructor of CharaceterAwareLM small version

    Args:
        char_size (int): size of characters
        vocab_size (int): size of vocabulary

    Returns:
        CharaceterAwareLM: CharaceterAwareLM small version
    """
    kernels = list(range(1, 7))
    return CharaceterAwareLM(
        char_size=char_size,
        vocab_size=vocab_size,
        embed_dim=15,
        hidden_dims=[25 * w for w in kernels],
        kernels=kernels,
        recurrent_hidden_dim=300,
        num_highway_layers=1,
        context=context,
    )


def characeter_aware_lm_large(
    char_size: int, vocab_size: int, context: NLPContext
) -> CharaceterAwareLM:
    """constructor of CharaceterAwareLM large version

    Args:
        char_size (int): size of characters
        vocab_size (int): size of vocabulary

    Returns:
        CharaceterAwareLM: CharaceterAwareLM large version
    """
    kernels = list(range(1, 8))
    return CharaceterAwareLM(
        char_size=char_size,
        vocab_size=vocab_size,
        embed_dim=15,
        hidden_dims=[min(200, 50 * w) for w in kernels],
        kernels=kernels,
        recurrent_hidden_dim=650,
        num_highway_layers=2,
        context=context,
    )
