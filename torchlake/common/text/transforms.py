from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from torchtext.vocab import Vocab

from . import functional as F

__all__ = [
    # "SentencePieceTokenizer",
    "VocabTransform",
    "ToTensor",
    # "LabelToIndex",
    "Truncate",
    "AddToken",
    "PadTransform",
    # "StrToIntTransform",
    # "GPT2BPETokenizer",
    # "CharBPETokenizer",
    # "RegexTokenizer",
    "Sequential",
]


class ToTensor(Module):
    r"""Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    """

    def __init__(
        self,
        padding_value: int | None = None,
        dtype: torch.dtype = torch.long,
    ) -> None:
        super().__init__()
        self.padding_value = padding_value
        self.dtype = dtype

    def forward(self, input: Any) -> Tensor:
        """
        :param input: Sequence or batch of token ids
        :type input: Union[List[int], List[List[int]]]
        :rtype: Tensor
        """
        return F.to_tensor(input, padding_value=self.padding_value, dtype=self.dtype)


class Sequential(torch.nn.Sequential):
    r"""A container to host a sequence of text transforms."""

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch. The input type must be supported by the first transform in the sequence.
        :type input: `Any`
        """
        for module in self:
            input = module(input)
        return input


class VocabTransform(Module):
    r"""Vocab transform to convert input batch of tokens into corresponding token ids

    :param vocab: an instance of :class:`torchtext.vocab.Vocab` class.

    Example:
        >>> import torch
        >>> from torchtext.vocab import vocab
        >>> from torchtext.transforms import VocabTransform
        >>> from collections import OrderedDict
        >>> vocab_obj = vocab(OrderedDict([('a', 1), ('b', 1), ('c', 1)]))
        >>> vocab_transform = VocabTransform(vocab_obj)
        >>> output = vocab_transform([['a','b'],['a','b','c']])
        >>> jit_vocab_transform = torch.jit.script(vocab_transform)
    """

    def __init__(self, vocab: Vocab) -> None:
        super().__init__()
        assert isinstance(vocab, Vocab)
        self.vocab = vocab

    def forward(self, input: Any) -> Any:
        """
        :param input: Input batch of token to convert to correspnding token ids
        :type input: Union[List[str], List[List[str]]]
        :return: Converted input into corresponding token ids
        :rtype: Union[List[int], List[List[int]]]
        """

        if torch.jit.isinstance(input, list[str]):
            return self.vocab.lookup_indices(input)
        elif torch.jit.isinstance(input, list[list[str]]):
            output: list[list[int]] = []
            for tokens in input:
                output.append(self.vocab.lookup_indices(tokens))

            return output
        else:
            raise TypeError("Input type not supported")


class Truncate(Module):
    r"""Truncate input sequence

    :param max_seq_len: The maximum allowable length for input sequence
    :type max_seq_len: int
    """

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch of sequence to be truncated
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        :return: Truncated sequence
        :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        return F.truncate(input, self.max_seq_len)


class AddToken(Module):
    """Add token to beginning or end of sequence

    :param token: The token to be added
    :type token: Union[int, str]
    :param begin: Whether to insert token at start or end or sequence, defaults to True
    :type begin: bool, optional
    """

    def __init__(self, token: int | str, begin: bool = True) -> None:
        super().__init__()
        self.token = token
        self.begin = begin

    def forward(self, input: Any) -> Any:
        """
        :param input: Input sequence or batch
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """

        return F.add_token(input, self.token, self.begin)


class PadTransform(Module):
    """Pad tensor to a fixed length with given padding value.

    :param max_length: Maximum length to pad to
    :type max_length: int
    :param pad_value: Value to pad the tensor with
    :type pad_value: bool
    """

    def __init__(self, max_length: int, pad_value: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: The tensor to pad
        :type x: Tensor
        :return: Tensor padded up to max_length with pad_value
        :rtype: Tensor
        """
        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x
