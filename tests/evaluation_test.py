import unittest
import numpy as np
import pytest
from src.evaluation import Evaluation

class EvaluationTest(unittest.TestCase):
    def test_perfect_predictions(self):
        y_actual = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        eval = Evaluation(y_pred, y_actual)
        assert eval.accuracy() == 1.0
        assert eval.precision() == 1.0
        assert eval.recall() == 1.0
    
    def test_all_wrong_predictions(self):
        y_actual = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        eval = Evaluation(y_pred, y_actual)
        assert eval.accuracy() == 0.0
    
    def test_shape_mismatch_raises(self):
        y_actual = np.array([0, 1])
        y_pred = np.array([0, 1, 1])
        with pytest.raises(ValueError):
            Evaluation(y_pred, y_actual)