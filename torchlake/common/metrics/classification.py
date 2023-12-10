import numpy as np


class IncrementalConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=int)

    def __str__(self) -> str:
        return str(self.matrix)

    def update(self, true_labels: list[int], predicted_labels: list[int]):
        self.matrix[true_labels, predicted_labels] += 1

    def get_confusion_matrix(self) -> np.ndarray:
        return self.matrix
