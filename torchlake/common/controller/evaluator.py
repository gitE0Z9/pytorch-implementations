from abc import ABC, abstractmethod
from typing import Iterator, Literal, TypeVar

import numpy as np
import torch
from matplotlib import pyplot as plt
from seaborn import heatmap
from torch import nn
from torchlake.common.metrics.classification import IncrementalConfusionMatrix
from torchmetrics import MeanSquaredError
from tqdm import tqdm

from ..mixins.controller import PredictFunctionMixin

T = TypeVar("T")


class EvaluatorBase(PredictFunctionMixin, ABC):
    @abstractmethod
    def _build_metric(self): ...

    @abstractmethod
    def _decode_output(
        self,
        output: torch.Tensor | tuple[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor]: ...

    def _update_metric(self, metric, y: torch.Tensor, yhat: torch.Tensor):
        metric.update(y.long(), yhat.detach().cpu().long())

    def run(self, data: Iterator, model: nn.Module, metric: T | None = None) -> T:
        if not hasattr(self, "_predict"):
            self.build_predict_function_by_data_type(iter(data))

        model.eval()
        with torch.no_grad():
            metric = metric or self._build_metric()
            for row in tqdm(data):
                _, y = row
                output = self._predict(row, model)
                output = self._decode_output(output, row=row)
                self._update_metric(metric, y, output)

        print(metric)
        return metric


class ClassificationEvaluator(EvaluatorBase):
    def __init__(
        self,
        label_size: int,
        device: torch.device,
        feature_dim: int | tuple[int] = 1,
        reduction: Literal["max"] | None = "max",
    ):
        """Evaluator for classification task

        Args:
            label_size (int): size of label
            device (torch.device): which device to use
            feature_dim (int | tuple[int], optional): which dimensions should be used as features. Defaults to 1.
            reduction (Literal["max"] | None, optional): how to decode feature dim. Defaults to None.
        """
        self.label_size = label_size
        self.device = device
        self.feature_dim = feature_dim
        self.reduction = reduction
        # TODO: ugly fix
        self.feature_last = False

    def _build_metric(self) -> IncrementalConfusionMatrix:
        return IncrementalConfusionMatrix(self.label_size)

    def _decode_output(
        self,
        output: torch.Tensor | tuple[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        if self.reduction == "max":
            return output.argmax(dim=self.feature_dim)
        else:
            return output

    # TODO: split classification matrix to itself utils, or just use torchmetric
    @staticmethod
    def get_matrix(
        confusion_matrix: IncrementalConfusionMatrix | np.ndarray,
    ) -> np.ndarray:
        matrix = confusion_matrix

        if isinstance(confusion_matrix, IncrementalConfusionMatrix):
            matrix = confusion_matrix.matrix

        return matrix

    def get_total_accuracy(
        self,
        confusion_matrix: IncrementalConfusionMatrix | np.ndarray,
    ) -> float:
        matrix = self.get_matrix(confusion_matrix)

        hits = matrix.diagonal().sum()
        total = matrix.sum()

        return np.where(total == 0, 0, hits / total)

    def get_per_class_accuracy(
        self,
        confusion_matrix: IncrementalConfusionMatrix | np.ndarray,
    ) -> np.ndarray:
        matrix = self.get_matrix(confusion_matrix)

        hits = matrix.diagonal()
        total = matrix.sum(axis=1)

        return np.where(total == 0, 0, hits / total)

    @staticmethod
    def show_per_class_accuracy(class_names: list[str], per_class_accs: list[float]):
        for class_name, class_acc in zip(class_names, per_class_accs):
            print(f"{class_name:<10}: {class_acc}")

    def plot_confusion_matrix(
        self,
        confusion_matrix: IncrementalConfusionMatrix | np.ndarray,
        labels: list[str],
        cmap: str | None = None,
        annot: bool = True,
        figsize: tuple[int] = (4, 3),
    ):
        matrix = self.get_matrix(confusion_matrix)

        hits = matrix
        total = matrix.sum(axis=1).reshape((self.label_size, 1))
        percentage = np.where(total == 0, 0, hits / total)

        plt.figure(figsize=figsize)
        heatmap(
            np.nan_to_num(percentage),
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=1,
            center=0.5,
            annot=annot,
            cmap=cmap,
        )


class RegressionEvaluator(EvaluatorBase):
    def __init__(
        self,
        device: torch.device,
        feature_dim: int | tuple[int] = 1,
    ):
        """Evaluator for classification task

        Args:
            device (torch.device): which device to use
            feature_dim (int | tuple[int], optional): which dimensions should be used as features. Defaults to 1.
        """
        self.device = device
        self.feature_dim = feature_dim
        # TODO: ugly fix
        self.feature_last = False

    def _build_metric(self):
        return MeanSquaredError()

    def _decode_output(
        self,
        output: torch.Tensor | tuple[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        return output
