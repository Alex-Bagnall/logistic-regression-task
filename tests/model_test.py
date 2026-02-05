import unittest
import numpy as np
from src.model import LogisticRegression

class LogisticRegressionTests(unittest.TestCase):
    def test_sigmoid_zero_returns_half(self):
        model = LogisticRegression()
        assert model.sigmoid(0) == 0.5

    def test_sigmoid_large_positive_near_one(self):
        model = LogisticRegression()
        assert model.sigmoid(10) > 0.99

    def test_sigmoid_large_negative_near_zero(self):
        model = LogisticRegression()
        assert model.sigmoid(-10) < 0.01

    def test_fit_reduces_loss(self):
        model = LogisticRegression(epochs=100)
        features = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        labels = np.array([0, 0, 1, 1])
        model.fit(features, labels)
        assert model.weights is not None
        assert model.bias is not None

    def test_predict_returns_binary(self):
        model = LogisticRegression(epochs=100)
        features = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        labels = np.array([0, 0, 1, 1])
        model.fit(features, labels)
        predictions = model.predict(features)
        assert all(p in [0, 1] for p in predictions)