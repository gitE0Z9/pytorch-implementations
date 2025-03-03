# copy from seq2seq

import torch
from torchlake.common.models.model_base import ModelBase
from torchlake.common.schemas.nlp import NlpContext
from torchlake.common.utils.sequence import get_input_sequence
from torchlake.sequence_data.models.base import RNNGenerator


class RNNLM(ModelBase):
    def __init__(
        self,
        backbone: RNNGenerator,
        context: NlpContext = NlpContext(),
    ):
        self.context = context
        super().__init__(
            None,
            None,
            head_kwargs={"backbone": backbone},
        )

    def build_foot(self, _, **kwargs):
        self.foot = ...

    def build_head(self, _, **kwargs):
        self.head: RNNGenerator = kwargs.pop("backbone")

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

    def loss_forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        early_stopping: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """training loss forward

        Args:
            x (torch.Tensor): groundtruth sequence, shape is (batch_size, seq_len)
            teacher_forcing_ratio (float, optional): scheduled sampling in paper [1506.03099]. Defaults to 0.5.
            early_stopping (bool, optional): stop generation after longest meaningful tokens in y. Defaults to True.

        Returns:
            torch.Tensor: generated sequence
        """
        return self.head.loss_forward(
            x,
            teacher_forcing_ratio=teacher_forcing_ratio,
            early_stopping=early_stopping,
        )

    def predict(
        self,
        x: torch.Tensor,
        topk: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """predict sequence with beam search

        Args:
            x (torch.Tensor): source sequence, shape in (batch, seq)
            topk (int, optional): beam search size. Defaults to 1.

        Returns:
            torch.Tensor: output sequence
        """
        batch_size = x.size(0)

        return self.head.predict(
            get_input_sequence((batch_size, 1), self.context),
            topk=topk,
        )
