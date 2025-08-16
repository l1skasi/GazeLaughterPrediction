from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    def __init__(self, X, y):
        """
        X_list: list of sequences, each sequence is a list of dicts (time steps)
        y_list: list of targets (e.g., class indices)
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
