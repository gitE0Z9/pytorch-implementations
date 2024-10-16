import torch
from torch import nn
from torchlake.common.models import FlattenFeature, KmaxPool1d
from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NlpContext

from .network import Block


class VDCNN(ModelBase):

    def __init__(
        self,
        vocab_size: int,
        output_size: int = 1,
        embed_dim: int = 16,
        topk: int = 8,
        depth_multipier: int = 1,
        enable_shortcut: bool = True,
        context: NlpContext = NlpContext(max_seq_len=1024),
    ):
        """Very deep convolution neural network in paper [1606.01781v2]

        Args:
            vocab_size (int): size of vocabulary
            output_size (int, optional): output channel size. Defaults to 1.
            embed_dim (int, optional): dimension of embedding vector. Defaults to 16.
            topk (int, optional): top k of max pooling. Defaults to 8.
            depth_multipier (int, optional): depth multiplier, 1 means 9 layers, 2 means 17 layers, and so on. Defaults to 1.
            enable_shortcut (bool, optional): enable shortcut. Defaults to True.
            context (NlpContext, optional): nlp context. Defaults to NlpContext(max_seq_len=1024).
        """
        self.embed_dim = embed_dim
        self.max_seq_len = context.max_seq_len
        self.topk = topk
        self.depth_multipier = depth_multipier
        self.enable_shortcut = enable_shortcut
        self.context = context
        super().__init__(vocab_size, output_size)  # dummy in_c

    @property
    def feature_dim(self) -> int:
        return 512 * self.topk

    @property
    def config(self) -> list[list[int]]:
        num_repeat = int(self.depth_multipier * 2)
        # in_c, out_c, num_repeat
        return [
            [64, 64, num_repeat],
            [64, 128, num_repeat],
            [128, 256, num_repeat],
            [256, 512, num_repeat],
        ]

    def build_foot(self, vocab_size: int):
        embed = nn.Embedding(vocab_size, self.embed_dim)
        conv = nn.Conv1d(self.embed_dim, 64, 3, padding=1)

        self.foot = nn.ModuleDict({"embed": embed, "conv": conv})

    def build_blocks(self):
        blocks = nn.Sequential()
        for stage_idx, (in_c, out_c, num_repeat) in enumerate(self.config):
            for layer_idx in range(num_repeat):
                _in_c = in_c if layer_idx == 0 else out_c
                _is_last_layer = layer_idx == num_repeat - 1
                _is_last_stage = stage_idx == len(self.config) - 1
                block = Block(
                    _in_c,
                    out_c,
                    3,
                    enable_shortcut=self.enable_shortcut
                    and (not (_is_last_layer and _is_last_stage)),
                    enable_pool=_is_last_layer and not _is_last_stage,
                )
                blocks.append(block)

        blocks.append(KmaxPool1d(self.topk))

        self.blocks = blocks

    def build_head(self, output_size: int):
        self.head = nn.Sequential(
            FlattenFeature(None, "1d"),
            nn.Linear(self.feature_dim, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.foot["embed"](x).transpose(-1, -2)
        y = self.foot["conv"](y)
        y = self.blocks(y)
        return self.head(y)
