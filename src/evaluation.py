import numpy as np
import logging

logger = logging.getLogger(__name__)

class Evaluation:
    def __init__(self, y_pred: np.ndarray, y_actual: np.ndarray):
        if y_pred.ndim != 1 or y_actual.ndim != 1:
            logger.error(
                "y_pred and y_actual must be 1D arrays, got %s and %s",
                y_pred.shape,
                y_actual.shape,
            )
            raise ValueError("y_pred and y_actual must be 1D arrays")

        if y_pred.shape != y_actual.shape:
            logger.error(
                "y_pred and y_actual shape mismatch: %s vs %s",
                y_pred.shape,
                y_actual.shape,
            )
            raise ValueError("y_pred and y_actual must have the same shape")

        if not np.all(np.isin(y_pred, (0, 1))) or not np.all(np.isin(y_actual, (0, 1))):
            logger.error(
                "Non-binary labels detected (y_pred unique=%s, y_actual unique=%s)",
                np.unique(y_pred),
                np.unique(y_actual),
            )
            raise ValueError("Labels must be binary (0 or 1)")

        self.y_pred = y_pred
        self.y_actual = y_actual
        self.true_positive: float = np.sum((y_pred == 1) & (y_actual == 1))
        self.false_positive: float = np.sum((y_pred == 1) & (y_actual == 0))
        self.true_negative: float = np.sum((y_pred == 0) & (y_actual == 0))
        self.false_negative: float = np.sum((y_pred == 0) & (y_actual == 1))

    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / len(self.y_actual)

    def precision(self) -> float:
        return self.true_positive / (self.true_positive + self.false_positive)

    def recall(self) -> float:
        return self.true_positive / (self.true_positive + self.false_negative)

    def f1(self) -> float:
        return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
