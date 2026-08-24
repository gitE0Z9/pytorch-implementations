import torch
from torch import nn
import torch.nn.functional as F

from torchlake.common.schemas.nlp import NLPContext
from torch.nn.functional import binary_cross_entropy_with_logits


class NCE(nn.Module):
    def __init__(
        self,
        word_freqs: torch.Tensor,
        negative_ratio: int = 5,
        power: float = 0.75,
        replacement: bool = True,
        exclude_padding: bool = False,
        context: NLPContext | None = None,
    ):
        """noise contrastive estimation

        Args:
            word_freqs (torch.Tensor): word frequency
            negative_ratio (int, optional): negative sample size compare to positive sample size. Defaults to 5.
            power (float, optional): power parameter. Defaults to 0.75.
            replacement (bool, optional): enable replacement for faster sampling. Defaluts to True.
            exclude_padding (bool, optional): remove padding from positive sample. Defaluts to False.
            context (NLPContext, optional): NLP context. Defaults to None.
        """
        assert negative_ratio > 0, "negative ratio should be higher than 0"

        if context is None:
            context = NLPContext()

        super().__init__()
        self.context = context
        self.negative_ratio = negative_ratio
        self.power = power
        self.distribution = self.get_distribution(word_freqs).to(context.device)
        self.vocab_size = self.distribution.numel()
        self.replacement = replacement
        self.exclude_padding = exclude_padding

    def get_distribution(self, word_freqs: torch.Tensor) -> torch.Tensor:
        """1310.4546 p.4
        noise distribution of word frequency formula

        Args:
            word_freqs (torch.Tensor): word frequency

        Returns:
            torch.Tensor: noise distribution, shape is (vocab_size)
        """
        return F.normalize(word_freqs.pow(self.power), p=1, dim=0)

    def sample(self, target: torch.Tensor) -> torch.Tensor:
        """negative sampling by noise distribution

        Args:
            target (torch.Tensor): shape(batch_size, 1 or neighbor_size, subseq)

        Returns:
            torch.Tensor: sampled token by noise distribution, shape is (batch_size, 1 or neighbor_size, subseq, #neg)
        """
        n: int = target.numel()
        output_shape = (
            target.size(0),
            target.size(1),
            target.size(2) * self.negative_ratio,
        )

        if self.replacement:
            # (n, #neg)
            y = self.distribution.multinomial(
                n * self.negative_ratio, replacement=True
            ).view(n, self.negative_ratio)

            collision = y == target.view(-1, 1)
            while collision.any():
                y[collision] = self.distribution.multinomial(
                    collision.sum().item(), replacement=True
                )
                collision = y == target.view(-1, 1)

            # (B, 1 or neighbor_size, subseq * #neg)
            return y.view(*output_shape)
        else:
            y = self.distribution.repeat(n, 1)
            # remove positive vocab
            # TODO: skipgram use target view as well
            # cbow could benefit from view but not skipgram
            y[torch.arange(n), target.reshape(-1)] = 0

            return (
                y
                # only 2 dim supported
                .multinomial(self.negative_ratio)
                # (B, 1 or neighbor_size, subseq * #neg)
                .view(*output_shape)
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
        # B, 1 or neighbor_size, subseq * #negative
        negative_x_indices = x_indices.repeat(1, 1, self.negative_ratio)
        # B, neighbor_size or 1, subseq * #negative
        negative_y_indices = self.sample(y_indices)
        # B, neighbor_size or 1, subseq * #negative
        negative_pred = model(negative_x_indices, negative_y_indices)

        positive_loss = binary_cross_entropy_with_logits(
            pred - self.negative_ratio * self.distribution[y_indices].log(),
            torch.ones_like(pred),
            reduction=(
                "mean"
                if not self.exclude_padding and self.context.padding_idx is None
                else "none"
            ),
        )

        negative_loss = binary_cross_entropy_with_logits(
            negative_pred
            - self.negative_ratio * self.distribution[negative_y_indices].log(),
            torch.zeros_like(negative_pred),
        )

        if self.exclude_padding and self.context.padding_idx is not None:
            positive_loss = positive_loss[y_indices != self.context.padding_idx].mean()

        return positive_loss + self.negative_ratio * negative_loss
