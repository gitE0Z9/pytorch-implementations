import torch
from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NlpContext
from torchlake.common.utils.sequence import get_input_sequence

from ..base import RNNDiscriminator, RNNGenerator


class Seq2Seq(ModelBase):
    def __init__(
        self,
        encoder: RNNDiscriminator,
        decoder: RNNGenerator,
        context: NlpContext = NlpContext(),
    ):
        self.context = context
        super().__init__(
            None,
            None,
            foot_kwargs={"encoder": encoder},
            head_kwargs={"decoder": decoder},
        )

    def build_foot(self, _, **kwargs):
        self.foot: RNNDiscriminator = kwargs.pop("encoder")

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """last hidden state of the encoder is used as the initial hidden state of the decoder

        Args:
            x (torch.Tensor): source sentence

        Returns:
            torch.Tensor: encoding
        """
        # 1409.3215 p.3: reverse x
        y = x.flip(1)
        # embedding
        y = self.foot.foot(y)
        # rnn feature
        return self.foot.feature_extract(x, y)

    def loss_forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        output_score: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """training loss forward

        Args:
            x (torch.Tensor): source sequence, shape is (batch_size, seq_len)
            y (torch.Tensor): groundtruth sequence, shape is (batch_size, seq_len)
            teacher_forcing_ratio (float, optional): scheduled sampling in paper [1506.03099]. Defaults to 0.5.
            output_score (bool, optional): output attention score or not. Defaults to False.

        Returns:
            torch.Tensor: generated sequence
        """
        # encoding
        o, hs, states = self.encode(x)

        # dL*ebi, B, S, eh
        # hs = (
        #     o[:, -1, :]
        #     .unflatten(-1, (self.foot.factor, -1))
        #     .permute(2, 0, 1, 3)
        #     .repeat(self.head.head.num_layers, 1, 1, 1)
        # )

        return self.head.loss_forward(
            y,
            hs,
            *states,
            ot=o,
            teacher_forcing_ratio=teacher_forcing_ratio,
            output_score=output_score,
        )

    def predict(
        self,
        x: torch.Tensor,
        topk: int = 1,
        output_score: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """predict sequence with beam search

        Args:
            x (torch.Tensor): source sequence, shape in (batch, seq)
            topk (int, optional): beam search size. Defaults to 1.
            output_score (bool, optional): output attention score or not. Defaults to False.

        Returns:
            torch.Tensor: output sequence
        """
        batch_size = x.size(0)

        # encoding
        o, hs, states = self.encode(x)

        return self.head.predict(
            get_input_sequence((batch_size, 1), self.context),
            hs,
            *states,
            ot=o,
            topk=topk,
            output_score=output_score,
        )
