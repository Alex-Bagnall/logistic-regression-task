import numpy as np
import logging

from evaluation import Evaluation
from src.preprocessing import Preprocessor
from tracking import ExperimentTracker

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.threshold = 0.5
        self.lambda_reg = 1

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, features, labels):
        n_samples, n_features = features.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for epoch in range(self.epochs):
            z = features.dot(self.weights) + self.bias
            eps = 1e-15
            prediction = np.clip(self.sigmoid(z), eps, 1 - eps)
            error = prediction - labels
            dw = ((1 / n_samples) * features.T.dot(error) + (self.lambda_reg / n_samples) * self.weights)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, features):
        z = features.dot(self.weights) + self.bias
        probabilities = self.sigmoid(z)
        return (probabilities >= self.threshold).astype(int)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = logging.getLogger(__name__)
    preprocessor = Preprocessor()
    features_train, features_test, labels_train, labels_test = preprocessor.preprocess()

    model = LogisticRegression()
    model.fit(features_train, labels_train)

    y_prediction = model.predict(features_test)
    prediction = Evaluation(y_prediction, labels_test)

    np.savez('../models/model.npz', weights=model.weights, bias=model.bias, mean=preprocessor.mean, std=preprocessor.std)

    tracker = ExperimentTracker()

    experiment = tracker.log_experiment(
        hyperparameters={
            "learning_rate": model.learning_rate,
            "epochs": model.epochs,
            "lambda_reg": model.lambda_reg,
            "threshold": model.threshold
        },
        metrics={
            "accuracy": prediction.accuracy(),
            "precision": prediction.precision(),
            "recall": prediction.recall(),
            "f1": prediction.f1()
        },
        model_path="model.npz"
    )