import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):

    def __init__(self, embeddings, labels, confidences, n_classes=None, transform=None, bn_shape=(3, 17, 17)):
        """

        Args:
            embeddings (torch.Tensor):
            labels (torch.Tensor):
            confidences (torch.Tensor):
            n_classes:
            transform:
        """
        if n_classes is None:
            n_classes = len(set(labels.tolist()))
        if confidences is None:
            confidences = torch.zeros(len(labels))

        # Normalize
        # with torch.no_grad():
        #    embedding_size = embeddings.shape[1]
        #    embeddings = embeddings.reshape(embeddings.shape[0], *bn_shape)
        #    embeddings = torch.nn.BatchNorm2d(3)(embeddings)
        #    embeddings = embeddings.reshape(embeddings.shape[0], embedding_size)

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

    '''
    def normalize_data(self, bn_shape=(3, 17, 17)):
        self.normalizer.train()
        with torch.no_grad():
            embedding_size = self.data.shape[1]
            embeddings = self.data.reshape(self.data.shape[0], *bn_shape)
            embeddings = self.normalizer(embeddings)
            self.data = embeddings.reshape(embeddings.shape[0], embedding_size)

    def normalize_samples(self, samples):
        with torch.no_grad():
            self.normalizer.eval()
            return self.normalizer(samples)
    '''
