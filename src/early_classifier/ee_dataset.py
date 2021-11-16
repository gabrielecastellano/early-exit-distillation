import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, embeddings, labels, n_classes=100, transform=None):
        self.data = embeddings[labels < n_classes]
        self.targets = labels[labels < n_classes]
        self.n_classes = n_classes
        self.transform = transform


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        embedding = self.data[idx]
        label = self.targets[idx]
        #target = torch.zeros(self.n_classes)
        #if label < self.n_classes:
        #    target[label] = 1

        sample = (embedding, label)

        if self.transform:
            sample = self.transform(sample)

        return sample