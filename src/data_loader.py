import csv
import numpy as np

class DataLoader:
    def __init__(self, file_path:str):
        self.file_path = file_path

    def load(self):
        label_map = {"Kecimen": 0, "Besni": 1}

        with open(self.file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

            data = []
            for row in reader:
                features = [float(x) for x in row[:-1]]
                label = label_map[row[-1]]
                data.append(features + [label])

        data = np.array(data)
        features = data[:, :-1]
        labels = data[:, -1]

        return features, labels