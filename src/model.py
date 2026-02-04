import numpy as np

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