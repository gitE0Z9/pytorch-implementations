from typing import Iterable

from torch import nn

from torchlake.common.controller.trainer import ClassificationTrainer


class SkipGramTrainer(ClassificationTrainer):
    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ):
        gram, context = row
        gram = gram.to(self.device)

        # batch, context-1, subseq, embed
        neighbor_size = context.size(1)

        return model(gram, neighbor_size, *args, **kwargs)
