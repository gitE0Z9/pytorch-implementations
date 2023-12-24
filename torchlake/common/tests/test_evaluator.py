import numpy as np

from ..controller.evaluator import ClassificationEvaluator
from ..metrics.classification import IncrementalConfusionMatrix

PREDICTIONS = [2, 1, 1, 2, 4, 0, 0]
LABELS = [1, 2, 3, 4, 0, 1, 0]

EXPECTED_MATRIX = np.array(
    [
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
    ]
)


def test_classification_evaluator_get_matrix():
    evaluator = ClassificationEvaluator(5, "cpu")

    matrix = IncrementalConfusionMatrix(5)
    evaluator.get_matrix(matrix)


def test_classification_evaluator_get_total_accuracy():
    evaluator = ClassificationEvaluator(5, "cpu")

    matrix = IncrementalConfusionMatrix(5)
    matrix.update(LABELS, PREDICTIONS)

    acc = evaluator.get_total_accuracy(matrix)

    assert acc == 1 / 7


def test_classification_evaluator_get_per_class_accuracy():
    evaluator = ClassificationEvaluator(5, "cpu")

    matrix = IncrementalConfusionMatrix(5)
    matrix.update(LABELS, PREDICTIONS)

    accs = evaluator.get_per_class_accuracy(matrix)

    print(accs)

    assert (
        np.testing.assert_equal(
            accs,
            np.array([1 / 2, 0, 0, 0, 0]),
        )
        is None
    )
