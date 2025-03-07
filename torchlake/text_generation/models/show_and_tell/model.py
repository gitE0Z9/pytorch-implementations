import torch
from torch import nn
from torchlake.common.models import FlattenFeature
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NlpContext
from torchlake.common.utils.sequence import get_input_sequence
from torchlake.sequence_data.models.base import RNNGenerator


class NeuralImageCation(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        decoder: RNNGenerator,
        context: NlpContext = NlpContext(),
        encode_dim: int | None = None,
    ):
        self.context = context
        super().__init__(
            None,
            None,
            foot_kwargs={"backbone": backbone},
            neck_kwargs={
                "encode_dim": encode_dim or backbone.feature_dims[-1],
                "embed_dim": decoder.head.embed_dim,
            },
            head_kwargs={"decoder": decoder},
        )

    def build_foot(self, _, **kwargs):
        self.foot = kwargs.pop("backbone")

    def build_neck(self, **kwargs):
        encode_dim = kwargs.pop("encode_dim")
        embed_dim = kwargs.pop("embed_dim")

        self.neck = nn.Sequential(
            FlattenFeature(),
        )

        if encode_dim != embed_dim:
            self.neck.append(
                nn.Linear(encode_dim, embed_dim),
            )

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
    ) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        # B, C, H, W
        z: torch.Tensor = self.foot(x).pop()

        # B, 1, C
        z: torch.Tensor = self.neck(z).unsqueeze(1)

        batch_size = z.size(0)
        # fed image feature as embedding
        # use fake input seq to avoid pack
        _, ht, states = self.head.head.feature_extract(
            get_input_sequence((batch_size, 1), self.context), z
        )

        return ht, states

    def loss_forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        output_score: bool = False,
    ) -> torch.Tensor:
        ht, states = self.encode(x)

        # B, S, V
        return self.head.loss_forward(
            y,
            ht,
            *states,
            teacher_forcing_ratio=teacher_forcing_ratio,
            output_score=output_score,
        )

    def predict(
        self,
        x: torch.Tensor,
        topk: int = 1,
        output_score: bool = False,
    ) -> torch.Tensor:
        ht, states = self.encode(x)

        # B, S
        x = get_input_sequence((x.size(0), 1), self.context)
        return self.head.predict(
            x,
            ht,
            *states,
            topk=topk,
            output_score=output_score,
        )
