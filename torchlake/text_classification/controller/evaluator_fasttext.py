from typing import Any, Iterable

import torch
from torch import nn


from torchlake.common.controller.evaluator import ClassificationEvaluator


class FastTextEvaluator(ClassificationEvaluator):
    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor | Any:
        (ngrams, words, word_spans), _ = row
        x = (
            [it.to(self.device) for it in ngrams],
            words.to(self.device),
            [it.to(self.device) for it in word_spans],
        )

        return model(*x, *args, **kwargs)
