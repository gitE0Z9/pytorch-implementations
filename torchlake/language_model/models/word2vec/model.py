import torch
import torch.nn.functional as F
from torch import nn

from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NLPContext
from torchlake.common.utils.tree import HuffmanNode
from torchlake.language_model.constants.enum import LossType, Word2VecModelType

from .network import HierarchicalSoftmax


class Word2Vec(ModelBase):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        model_type: Word2VecModelType,
        loss_type: LossType = LossType.CROSS_ENTROPY,
        context: NLPContext | None = None,
        root: HuffmanNode | None = None,
    ):
        """Word2Vec model

        Args:
            vocab_size (int): vocabulary size
            embed_dim (int): embedding dimension
            model_type (Word2VecModelType): model type, either CBOW or SkipGram
            loss_type (LossType, optional): loss type, cross entropy, negative sampling, hierarchical softmax. Defaults to LossType.CROSS_ENTROPY.
            context (NLPContext, optional): NLP context. Defaults to None.
            root (HuffmanNode, optional): the root of a Huffman tree based on word counts. Defaults to None.
        """
        if context is None:
            context = NLPContext()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context = context
        self.model_type = model_type
        self.loss_type = loss_type
        super().__init__(
            vocab_size,
            vocab_size,
            head_kwargs={
                "root": root,
            },
        )

    @property
    def input_embedding(self) -> nn.Embedding:
        return self.foot

    @property
    def output_embedding(self) -> nn.Embedding:
        if self.loss_type == LossType.CROSS_ENTROPY:
            w = self.head.weight.detach()
            out_c, in_c = w.shape
            e = nn.Embedding(
                out_c,
                in_c,
                padding_idx=self.context.padding_idx,
            )
            e.weight.data.copy_(w)
            return e
        elif self.loss_type == LossType.NEGATIVE_SAMPLING:
            return self.head
        else:
            raise ValueError("only supports when loss type is ce or ns")

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Embedding(
            input_channel,
            self.embed_dim,
            padding_idx=self.context.padding_idx,
        )

    def build_blocks(self, **kwargs):
        if self.model_type == Word2VecModelType.CBOW:
            # aggregate context with "mean" as author(https://github.com/tmikolov/word2vec/blob/master/word2vec.c)
            # "sum" used in paper(https://arxiv.org/abs/1301.3781) is deprecated
            # b*subseq, 1, h
            self.blocks = lambda x, _: x.mean(1, keepdim=True)
        elif self.model_type == Word2VecModelType.SKIP_GRAM:
            # b*subseq, n, h
            self.blocks = lambda x, neighbor_size: x.repeat(1, neighbor_size, 1)
        else:
            raise ValueError("only supports when model type is cbow or skip gram")

    def build_head(self, _, **kwargs):
        if self.loss_type == LossType.CROSS_ENTROPY:
            self.head = nn.Linear(
                self.embed_dim,
                self.vocab_size,
                # bias=False,
            )
        elif self.loss_type == LossType.NEGATIVE_SAMPLING:
            self.head = nn.Embedding(
                self.vocab_size,
                self.embed_dim,
                padding_idx=self.context.padding_idx,
            )
        elif self.loss_type == LossType.HIERARCHICAL_SOFTMAX:
            root = kwargs.pop("root")
            self.head = HierarchicalSoftmax(
                root,
                self.vocab_size,
                self.embed_dim,
                context=self.context,
            )
        else:
            raise ValueError

    def forward(self, x: torch.Tensor, neighbor_size: int = 1) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): one-hot vector of tokens, shape is (batch_size*#subsequence, 1 or neighbor_size)
            neighbor_size (int): context size for skipgram model. Defaults to 1.

        Returns:
            torch.Tensor: output vector, shape is (batch_size*#subsequence, neighbor_size or 1, embedding_dim)
        """
        # cb: b*subseq, n, h # sg: b*subseq, 1, h
        y = self.foot(x)
        # cb: b*subseq, 1, h # sg: b*subseq, n, h
        # neighbor_size is specified for random context
        # otherwise it should be fixed
        y = self.blocks(y, neighbor_size)

        if (self.loss_type == LossType.CROSS_ENTROPY) or (
            self.loss_type == LossType.HIERARCHICAL_SOFTMAX and not self.training
        ):
            # ce: b*subseq, n or 1, vocab_size
            # hs: b*subseq, n or 1
            return self.head(y)
        elif self.loss_type == LossType.NEGATIVE_SAMPLING and not self.training:
            # b*subseq, n or 1, vocab_size
            return F.linear(y, self.head.weight)

        return y
