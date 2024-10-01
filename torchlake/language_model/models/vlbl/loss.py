import torch
from torch import nn

from torchlake.common.schemas.nlp import NlpContext
from torch.nn.functional import binary_cross_entropy_with_logits


class NCE(nn.Module):
    def __init__(
        self,
        word_freqs: torch.Tensor,
        negative_ratio: int = 5,
        power: float = 0.75,
        context: NlpContext = NlpContext(),
    ):
        """noise contrastive estimation

        Args:
            word_freqs (torch.Tensor): word frequency
            negative_ratio (int, optional): negative sample size compare to positive sample size. Defaults to 5.
            power (float, optional): power parameter. Defaults to 0.75.
            context (NlpContext, optional): context object. Defaults to NlpContext().
        """
        super(NCE, self).__init__()
        self.context = context
        self.negative_ratio = negative_ratio
        self.power = power
        self.distribution = self.get_distribution(word_freqs).to(context.device)
        self.vocab_size = self.distribution.numel()

        assert negative_ratio > 0, "negative ratio should be higher than 0"

    def get_distribution(self, word_freqs: torch.Tensor) -> torch.Tensor:
        """1310.4546 p.4
        noise distribution of word frequency formula

        Args:
            word_freqs (torch.Tensor): word frequency

        Returns:
            torch.Tensor: noise distribution, shape is (vocab_size)
        """
        return nn.functional.normalize(word_freqs.pow(self.power), p=1, dim=0)

    def sample(self, target: torch.Tensor) -> torch.Tensor:
        """negative sampling by noise distribution

        Args:
            target (torch.Tensor): shape(batch_size, 1 or neighbor_size, #subsequence)

        Returns:
            torch.Tensor: sampled token by noise distribution, shape is (B, context-1, subseq * #neg)
        """
        n: int = target.numel()
        y = self.distribution.repeat(n, 1)
        # remove positive vocab
        # TODO: skipgram use target view as well
        # cbow could benefit from view but not skipgram
        y[torch.arange(n), target.reshape(-1)] = 0

        return (
            y
            # only 2 dim supported
            .multinomial(self.negative_ratio)
            # (B, context-1, subseq, neg)
            .view(target.size(0), target.size(1), target.size(2) * self.negative_ratio)
        )

    def forward(
        self,
        model: nn.Module,
        x_indices: torch.Tensor,
        y_indices: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        """compute noise contrastive estimation loss

        Args:
            model (nn.Module): VLBL or IVLBL
            x_indices (torch.Tensor): shape(batch_size, 1 or neighbor_size, #subsequence)
            y_indices (torch.Tensor): shape(batch_size, neighbor_size or 1, #subsequence)
            pred (torch.Tensor): shape(batch_size, neighbor_size or 1, #subsequence)

        Returns:
            torch.Tensor: nce loss value
        """
        # B, 1 or neighbor, subseq*#negative
        negative_x_indices = x_indices.repeat(1, 1, self.negative_ratio)
        # B, neighbor or 1, subseq*#negative
        negative_y_indices = self.sample(y_indices)
        # B, neighbor or 1, subseq*#negative
        negative_pred = model.forward(negative_x_indices, negative_y_indices)

        positive_loss = binary_cross_entropy_with_logits(
            pred - self.negative_ratio * self.distribution[y_indices].log(),
            torch.ones_like(pred),
        )

        negative_loss = binary_cross_entropy_with_logits(
            negative_pred
            - self.negative_ratio * self.distribution[negative_y_indices].log(),
            torch.zeros_like(negative_pred),
        )

        return positive_loss + self.negative_ratio * negative_loss
