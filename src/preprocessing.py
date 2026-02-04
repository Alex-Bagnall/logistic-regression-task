import numpy as np

from src.data_loader import DataLoader

class Preprocessor:
    def __init__(self):
        self.std = None
        self.mean = None

    def create_normalised_array(self, features):
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        return (features - self.mean) / self.std

    def preprocess(self, seed):
        data_loader = DataLoader("../data/raisin_dataset.csv")
        features, labels = data_loader.load()
        normalised_features = self.create_normalised_array(features)

        np.random.seed(seed)
        indices = np.random.permutation(len(normalised_features))
        split = int(len(normalised_features) * 0.8)

        features_train = normalised_features[indices[:split]]
        features_test = normalised_features[indices[split:]]
        labels_train = labels[indices[:split]]
        labels_test = labels[indices[split:]]
        return features_train, features_test, labels_train, labels_test

if __name__ == "__main__":
    loader = Preprocessor()
    loader.preprocess(42)