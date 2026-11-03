from typing import Literal, Sequence

import torch
from torch import nn

from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NLPContext
from .network import glu_kaming_init
from .network import Bottleneck, ResBlock


class GatedConvLM(ModelBase):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        key: Literal["8", "9", "13", "14", "8B", "14B"] | None = None,
        stage_configs: tuple[tuple[int, tuple[int], tuple[int]]] | None = None,
        head_type: Literal["conv", "linear", "adaptive_softmax"] = "conv",
        cutoffs: Sequence[int] | None = None,
        div_value: float = 4,
        preactivation: bool = False,
        context: NLPContext | None = None,
    ):
        """Gated convolution language model [1612.08083v3]

        Args:
            vocab_size (int): _description_
            embed_dim (int): _description_
            output_size (int): _description_
            key (Literal["8", "9", "13", "14", "8B", "14B"] | None, optional): _description_. Defaults to None.
            stage_configs (tuple[tuple[int, tuple[int], tuple[int]]] | None, optional): format is num_block, hidden_dims, kernel_sizes. Defaults to None.
            head_type (Literal["conv", "linear", "adaptive_softmax"] | None, optional): _description_. Defaults to "conv".
            cutoffs (Sequence[int] | None, optional): _description_. Defaults to None.
            div_value (float, optional): _description_. Defaults to 4.
            preactivation (bool, optional): _description_. Defaults to False.
            context (NLPContext | None, optional): _description_. Defaults to None.
        """
        assert (
            key is not None or stage_configs is not None
        ), "must provide either key or stage_config"

        if context is None:
            context = NLPContext()

        self.embed_dim = embed_dim

        if key is not None:
            self.stage_configs = self.config[key]
        else:
            self.stage_configs = stage_configs

        self.head_type = head_type
        self.preactivation = preactivation
        self.context = context
        super().__init__(
            vocab_size,
            vocab_size,
            head_kwargs={
                "cutoffs": cutoffs,
                "div_value": div_value,
            },
        )

    @property
    def feature_dim(self) -> int:
        _, hidden_dims, _ = self.stage_configs[-1]
        return hidden_dims[-1]

    @property
    def config(self) -> dict[str, tuple[tuple[int, tuple[int], tuple[int]]]]:
        # format is num_block, hidden_dims, kernels
        return {
            "8": (
                (1, (900,), (4,)),
                (7, (900,), (4,)),
            ),
            "9": (
                (1, (807,), (4,)),
                (4, (807, 807), (4, 4)),
            ),
            "13": (
                (1, (1268,), (4,)),
                (12, (1268, 1268), (4, 4)),
            ),
            "14": (
                (3, (850,), (6,)),
                (1, (850,), (1,)),
                (4, (850,), (5,)),
                (1, (850,), (1,)),
                (3, (850,), (4,)),
                (1, (1024,), (4,)),
                (1, (2048,), (4,)),
            ),
            "8B": (
                (1, (512,), (1,)),
                (3, (128, 128, 512), (1, 5, 1)),
                (3, (256, 256, 512), (1, 5, 1)),
                (1, (1024, 1024, 2048), (1, 5, 1)),
            ),
            "14B": (
                (1, (512,), (5,)),
                (3, (128, 128, 512), (1, 5, 1)),
                (3, (512, 512, 1024), (1, 5, 1)),
                (6, (1024, 1024, 2048), (1, 5, 1)),
                (1, (1024, 1024, 4096), (1, 5, 1)),
            ),
        }

    @property
    def input_embedding(self) -> nn.Embedding:
        return self.foot

    @property
    def output_embedding(self) -> nn.Embedding:
        if self.head_type in ("conv", "linear"):
            w = self.head[0].weight.detach()
            out_c, in_c = w.shape[:2]
            e = nn.Embedding(
                out_c,
                in_c,
                padding_idx=self.context.padding_idx,
            )
            e.weight.data.copy_(w.squeeze())
        else:
            raise ValueError("only support conv or linear head")

        return e

    def build_foot(self, input_channel, **kwargs):
        self.foot = nn.Embedding(
            input_channel,
            self.embed_dim,
            padding_idx=self.context.padding_idx,
        )

    def build_blocks(self, **kwargs):
        self.blocks = nn.Sequential()

        in_c = self.embed_dim
        for stage_cfg in self.stage_configs:
            num_block, hidden_dims, kernels = stage_cfg
            out_c = hidden_dims[-1]
            for _ in range(num_block):
                self.blocks.append(
                    ResBlock(
                        in_c,
                        out_c,
                        block=Bottleneck(
                            in_c,
                            hidden_dims=hidden_dims,
                            kernels=kernels,
                            preactivation=self.preactivation,
                        ),
                    )
                )
                in_c = out_c

    def build_head(self, output_size, **kwargs):
        # consumes 1GB more vram for 9% training speed gain
        if self.head_type == "conv":
            layer = nn.Conv1d(self.feature_dim, output_size, 1)
            glu_kaming_init(layer.weight, layer.bias)
        elif self.head_type == "linear":
            # transpose might be the reason to be slower
            layer = nn.Linear(self.feature_dim, output_size)
            nn.init.kaiming_normal_(layer.weight, 0, nonlinearity="linear")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif self.head_type == "adaptive_softmax":
            cutoffs = kwargs.pop("cutoffs")
            div_value = kwargs.pop("div_value", 4)
            layer = nn.AdaptiveLogSoftmaxWithLoss(
                self.feature_dim,
                output_size,
                cutoffs=cutoffs,
                div_value=div_value,
            )

        self.head = nn.Sequential(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.foot(x)
        y = self.blocks(y.transpose(1, 2))

        if self.head_type == "linear":
            y = y.transpose(1, 2)
        elif self.head_type == "adaptive_softmax":
            if self.training:
                return self.head(y).loss
            else:
                return self.head.log_prob(y)

        return self.head(y)
