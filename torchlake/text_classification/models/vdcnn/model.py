import torch
from torch import nn
from torchlake.common.models.cnn_base import ModelBase
from torchlake.common.schemas.nlp import NlpContext

from .network import Block, KmaxPool1d


class Vdcnn(ModelBase):

    def __init__(
        self,
        vocab_size: int,
        output_size: int = 1,
        embed_dim: int = 16,
        topk: int = 8,
        depth_multipier: int = 1,
        context: NlpContext = NlpContext(max_seq_len=1024),
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = context.max_seq_len
        self.topk = topk
        self.depth_multipier = depth_multipier
        self.context = context
        super(Vdcnn, self).__init__(1, output_size)  # dummy in_c

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

    def build_foot(self, input_channel: int):
        embed = nn.Embedding(self.vocab_size, self.embed_dim)
        conv = nn.Conv1d(self.embed_dim, 64, 3, padding=1)

        self.foot = nn.ModuleDict({"embed": embed, "conv": conv})

    def build_blocks(self):
        blocks = nn.Sequential()
        for stage_idx, (in_c, out_c, num_repeat) in enumerate(self.config):
            for layer_idx in range(num_repeat):
                block = Block(in_c if layer_idx == 0 else out_c, out_c, 3)
                blocks.append(block)

                if layer_idx == num_repeat - 1 and stage_idx != len(self.config) - 1:
                    blocks.append(nn.MaxPool1d(3, 2, 1))

        blocks.append(KmaxPool1d(self.topk))

        self.blocks = blocks

    def build_head(self, output_size: int):
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.foot["embed"](x).transpose(-1, -2)
        y = self.foot["conv"](y)
        y = self.blocks(y).view(x.size(0), -1)
        return self.head(y)
