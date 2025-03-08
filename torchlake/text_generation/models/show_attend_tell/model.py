import torch
from torch import nn
from torchlake.common.models import FlattenFeature
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NlpContext
from torchlake.common.utils.sequence import get_input_sequence
from torchlake.sequence_data.models.base.rnn_generator import RNNGenerator


class ShowAttendTell(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        decoder: RNNGenerator,
        context: NlpContext = NlpContext(),
        encode_dim: int | None = None,
    ):
        self.context = context

        multiple_state = isinstance(decoder.head.blocks, nn.LSTM)

        super().__init__(
            None,
            None,
            foot_kwargs={"backbone": backbone},
            neck_kwargs={
                "multiple_state": multiple_state,
                "encode_dim": encode_dim or backbone.feature_dims[-1],
                "decode_dim": decoder.head.hidden_dim,
            },
            head_kwargs={"decoder": decoder},
        )

    def build_foot(self, _, **kwargs):
        self.foot = kwargs.pop("backbone")

    def build_neck(self, **kwargs):
        encode_dim = kwargs.pop("encode_dim")
        decode_dim = kwargs.pop("decode_dim")

        self.neck = nn.ModuleDict(
            {
                "flatten": FlattenFeature(),
                "h": nn.Linear(encode_dim, decode_dim),
            }
        )

        if kwargs.pop("multiple_state"):
            self.neck["c"] = nn.Linear(encode_dim, decode_dim)

    def build_head(self, _, **kwargs):
        self.head: RNNGenerator = kwargs.pop("decoder")

    def train(self, mode=True):
        result = super().train(mode)
        result.head.train()
        result.forward = self.loss_forward
        return result

    def eval(self):
        result = super().eval()
        result.head.eval()
        result.forward = self.predict
        return result

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor]]:
        # B, C, H, W
        a: torch.Tensor = self.foot(x).pop()

        # 1, B, C
        z: torch.Tensor = self.neck["flatten"](a).unsqueeze(0)
        D = self.head.head.num_layers * self.head.head.factor
        ht = self.neck["h"](z)
        states = tuple()
        if "c" in self.neck:
            states = tuple(self.neck["c"](z))

        # B, H*W, C # D, B, h # D, B, h
        return (
            a.flatten(2).transpose(-1, -2),
            ht.repeat(D, 1, 1),
            tuple(state.repeat(D, 1, 1) for state in states),
        )

    def loss_forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        output_score: bool = True,
        early_stopping: bool = True,
    ) -> torch.Tensor:
        o, ht, states = self.encode(x)

        # B, S, V
        return self.head.loss_forward(
            y,
            ht,
            *states,
            ot=o,
            teacher_forcing_ratio=teacher_forcing_ratio,
            output_score=output_score,
            early_stopping=early_stopping,
        )

    def predict(
        self,
        x: torch.Tensor,
        topk: int = 1,
        output_score: bool = False,
    ) -> torch.Tensor:
        o, ht, states = self.encode(x)

        # B, S
        x = get_input_sequence((x.size(0), 1), self.context)
        return self.head.predict(
            x,
            ht,
            *states,
            ot=o,
            topk=topk,
            output_score=output_score,
        )
