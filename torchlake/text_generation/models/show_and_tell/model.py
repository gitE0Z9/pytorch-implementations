import torch
from torchlake.common.models import FlattenFeature
from torchlake.common.models.feature_extractor_base import ExtractorBase
from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NlpContext
from torchlake.sequence_data.models.base import RNNGenerator
from torchlake.sequence_data.models.lstm import LSTMDiscriminator
from torchlake.common.utils.sequence import get_input_sequence


class NeuralImageCation(ModelBase):

    def __init__(
        self,
        backbone: ExtractorBase,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_size: int = 1,
        num_layers: int = 1,
        bidirectional: bool = False,
        context: NlpContext = NlpContext(),
    ):
        input_channel = 3
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.context = context
        super().__init__(
            input_channel,
            output_size,
            foot_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot = kwargs.pop("backbone")

    def build_neck(self):
        self.neck = FlattenFeature()

    def build_head(self, output_size: int):
        model = LSTMDiscriminator(
            self.vocab_size,
            self.embed_dim,
            self.hidden_dim,
            output_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            context=self.context,
        )

        self.head = RNNGenerator(model)

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

    def get_image_feature(
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

    def loss_forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ht, states = self.get_image_feature(x)

        # B, S, V
        return self.head.loss_forward(y, ht, *states)

    def predict(self, x: torch.Tensor, topk: int = 1) -> torch.Tensor:
        ht, states = self.get_image_feature(x)

        # B, S
        x = get_input_sequence((x.size(0), 1), self.context)
        return self.head.predict(x, ht, *states, topk=topk)
