from typing import Sequence

import numpy as np


class IncrementalConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=int)

    def __str__(self) -> str:
        return str(self.matrix)

    def update(self, pred: Sequence[int], gt: Sequence[int]):
        for true, predicted in zip(gt, pred):
            self.matrix[true, predicted] += 1

    def get_confusion_matrix(self) -> np.ndarray:
        return self.matrix
