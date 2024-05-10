from typing import Iterator
from matplotlib import pyplot as plt

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

            print(confusion_matrix)

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
