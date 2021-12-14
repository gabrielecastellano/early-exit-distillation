import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):

    def __init__(self, embeddings, labels, confidences, n_classes=None, transform=None):
        """

        Args:
            embeddings (torch.Tensor):
            labels (torch.Tensor):
            confidences (torch.Tensor):
            n_classes:
            transform:
        """
        if n_classes is None:
            n_classes = len(labels)
        if confidences is None:
            confidences = torch.zeros(len(labels))
        self.data = embeddings[labels < n_classes]
        self.targets = labels[labels < n_classes]
        self.confidences = confidences[labels < n_classes]
        self.transform = transform


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        embedding = self.data[idx]
        label = self.targets[idx]

        sample = (embedding, label)

        if self.transform:
            sample = self.transform(sample)

        return sample
