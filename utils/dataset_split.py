import numpy as np

class DatasetSplit:
    def __init__(self, dataset, n_parts=4, train_size=0.5, val_size=0.25):

        self.n_parts = n_parts
        self.train_size = train_size
        self.val_size = val_size

        self.dataset = dataset
        self.n = len(dataset)
        self.part_size = self.n // n_parts

    def split_dataset(self):
        """
        Split dataset into train, validation, and test sets.

        """
        train_idx, val_idx, test_idx = [], [], []

        for i in range(self.n_parts):
            # Determine the index range
            start = i * self.part_size
            end = (i + 1) * self.part_size if i < self.n_parts - 1 else self.n

            part_indices = np.arange(start, end)

            # Number of samples for train and validation
            n_train = int(len(part_indices) * self.train_size)
            n_val = int(len(part_indices) * self.val_size)

            train_idx.extend(part_indices[:n_train])
            val_idx.extend(part_indices[n_train:n_train + n_val])
            test_idx.extend(part_indices[n_train + n_val:])

        X_train = [self.dataset[i] for i in train_idx]
        X_validation = [self.dataset[i] for i in val_idx]
        X_test = [self.dataset[i] for i in test_idx]
        return X_train, X_validation, X_test

    def get_features_and_targets(self, dataset):
        """
        Separate features (X) and targets (y) for a given dataset subset.

        Targets are the values corresponding to specific keys:
        - 'Laughter@CHI'
        - 'Gaze@CHI'
        - 'GazeRelation'
        """
        X = []
        y = []

        for record in dataset:
            target = {}
            input = {}

            for key, value in record.items():
                if key in ['Laughter@CHI', 'Gaze@CHI', 'GazeRelation']:
                    target[key] = value
                else:
                    input[key] = value

            X.append(input)
            y.append(target)

        return X, y

    def main(self):
        train, validation, test = self.split_dataset()

        X_train, y_train = self.get_features_and_targets(train)
        X_validation, y_validation = self.get_features_and_targets(validation)
        X_test, y_test = self.get_features_and_targets(test)

        return X_train, y_train, X_validation, y_validation, X_test, y_test
