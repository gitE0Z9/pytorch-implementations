from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

import torch
from torch import nn


class ModelBase(nn.Module, ABC):

    def __init__(
        self,
        input_channel: int,
        output_size: int,
    ):
        super().__init__()
        self.build_encoder(input_channel)
        self.build_decoder(output_size)
        self.build_neck()
        self.build_head(output_size)

    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    @property
    def config(self) -> list[list[Any]]:
        raise NotImplementedError

    @abstractmethod
    def build_encoder(self, input_channel: int):
        self.encoder = ...

    @abstractmethod
    def build_decoder(self, output_size: int):
        self.decoder = ...

    def build_neck(self):
        self.neck = None

    @abstractmethod
    def build_head(self, output_size: int):
        self.head = ...

    @abstractmethod
    def encode(self): ...

    @abstractmethod
    def decode(self): ...

    def forward(
        self,
        encode_x: torch.Tensor,
        decode_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encode_y = self.encoder(encode_x)
        y = self.decoder(decode_x, encode_y)

        if self.neck is not None:
            y = self.neck(y)

        return self.head(y)


T = TypeVar("T")
U = TypeVar("U")


class EncoderDecoderModel(Generic[T, U], nn.Module):
    def __init__(
        self,
        encoder: T,
        decoder: U,
        neck_kwargs: dict | None = {},
        branch_kwargs: dict | None = {},
        head_kwargs: dict | None = {},
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.build_neck(**neck_kwargs)
        self.build_branch(**branch_kwargs)
        self.build_head(**head_kwargs)

    def build_neck(self, *args, **kwargs):
        self.neck = None

    def build_branch(self, *args, **kwargs):
        self.branch = None

    def build_head(self, *args, **kwargs):
        self.head = None
