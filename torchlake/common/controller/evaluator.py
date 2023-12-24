from typing import Iterator

import numpy as np
import torch
from seaborn import heatmap
from torch import nn
from torchlake.common.metrics.classification import IncrementalConfusionMatrix
from tqdm import tqdm


class ClassificationEvaluator:
    def __init__(self, label_size: int, device: torch.device):
        self.label_size = label_size
        self.device = device

    def run(self, data: Iterator, model: nn.Module):
        model.eval()
        with torch.no_grad():
            confusion_matrix = IncrementalConfusionMatrix(self.label_size)

            for x, y in tqdm(data):
                x = x.to(self.device)

                output = model(x).argmax(dim=-1)
                confusion_matrix.update(y.long(), output.detach().cpu())

            print(self.get_total_accuracy(confusion_matrix))

        return confusion_matrix

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

        acc = matrix.diagonal().sum() / matrix.sum()

        return acc

    def get_per_class_accuracy(
        self,
        confusion_matrix: IncrementalConfusionMatrix | np.ndarray,
    ) -> np.ndarray:
        matrix = self.get_matrix(confusion_matrix)

        accs = matrix.diagonal() / matrix.sum(axis=1)

        return accs

    @staticmethod
    def show_per_class_accuracy(class_names: list[str], per_class_accs: list[float]):
        for class_name, class_acc in zip(class_names, per_class_accs):
            print(f"{class_name:<10}: {class_acc}")

    def plot_confusion_matrix(
        self,
        confusion_matrix: IncrementalConfusionMatrix | np.ndarray,
        labels: list[str],
        cmap: str | None = None,
    ):
        matrix = self.get_matrix(confusion_matrix)

        percentage = matrix / matrix.sum(axis=1).reshape((self.label_size, 1))

        heatmap(
            percentage,
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=1,
            center=0.5,
            annot=True,
            cmap=cmap,
        )
