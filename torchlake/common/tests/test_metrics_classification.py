import numpy as np
import torch
from ..metrics.classification import IncrementalConfusionMatrix

PREDICTIONS = [1, 1, 1, 2, 4, 0, 0]
LABELS = [1, 1, 3, 4, 0, 1, 0]

EXPECTED_MATRIX = np.array(
    [
        [1, 0, 0, 0, 1],
        [1, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ]
)


def test_incremental_confusion_matrix_update():
    confusion_matrix = IncrementalConfusionMatrix(5)

    confusion_matrix.update(LABELS, PREDICTIONS)

    assert np.testing.assert_equal(confusion_matrix.matrix, EXPECTED_MATRIX) is None
