import numpy as np
import torch
import faiss

from early_classifier.base import BaseClassifier
from early_classifier.ee_dataset import EmbeddingDataset
from utils import dataset_util


class FaissKMeansClassifier(BaseClassifier):


    def __init__(self, device, n_labels, k, dim, niter=10, threshold=0):
        super().__init__(device, n_labels)
        self.k = k
        self.dim = dim
        self.niter = niter
        self.model = faiss.Kmeans(self.dim, self.k, niter=self.niter, gpu='cuda' in device.type)
        self.label_share = np.zeros([self.k, n_labels], dtype=int)
        self.confidence_share = {cluster: {label: [] for label in range(n_labels)} for cluster in range(self.k)}
        self.max_labels = np.zeros(self.k, dtype=int)
        self.shares = np.zeros(self.k, dtype=float)
        self.cluster_sizes = np.zeros(self.k, dtype=int)
        self.valid_shares = None
        self.share_threshold = threshold
        if type(threshold) == list:
            self.share_threshold = threshold[0]
        self.share_threshold = self.share_threshold if self.share_threshold != 'auto' else 0.5
        self.distances_q = None
        self.dataset = None

    def fit(self, data_loader, epoch=0):
        """

        Args:
            data_loader (DataLoader):
            epoch:

        """
        if type(data_loader.dataset) != EmbeddingDataset:
            raise TypeError(f"Unexpected dataset type '{type(data_loader.dataset)}'")

        self.dataset = data_loader.dataset

        x = data_loader.dataset.data
        y = data_loader.dataset.targets
        c = data_loader.dataset.confidences

        # clustering
        self.model.train(x)

        # predict on the training set
        centroid_distances, clusters = self.model.assign(x)
        norm_distances = centroid_distances / centroid_distances.max()
        self.distances_q = np.percentile(norm_distances, (0, 0.25, 0.5, 0.75, 1))

        # compute per-cluster shares
        self.confidence_share = {cluster: {label: [] for label in range(self.n_labels)} for cluster in range(self.k)}
        for cluster, target, confidence in zip(clusters, y, c):
            if cluster != -1:
                if target < self.n_labels:
                    self.label_share[cluster][target] += 1
                    self.confidence_share[cluster][target].append(confidence)
        for cluster in self.confidence_share:
            for label in self.confidence_share[cluster]:
                if len(self.confidence_share[cluster][label]) > 0:
                    self.confidence_share[cluster][label] = np.mean(self.confidence_share[cluster][label])
                else:
                    self.confidence_share[cluster][label] = 0
        for cluster in range(self.k):
            self.max_labels[cluster] = np.argmax(self.label_share[cluster])
            if self.label_share[cluster][self.max_labels[cluster]] > 0:
                self.shares[cluster] = np.max(self.label_share[cluster]) / np.sum(self.label_share[cluster])
            self.cluster_sizes[cluster] = np.sum(self.label_share[cluster])

        # only keep clusters featuring more than one item
        self.valid_shares = [share for i, share in enumerate(self.shares) if self.cluster_sizes[i] > 1]

    def predict(self, x):

        # TODO is it possible to work directly with tensors?
        x = x.cpu().detach().numpy()
        y = np.zeros((x.shape[0], self.n_labels))
        centroid_distances, predicted_clusters = self.model.assign(x)
        norm_distances = centroid_distances / centroid_distances.max()

        for i, cluster in enumerate(predicted_clusters):
            if cluster != -1:
                # Naive confidence: fraction of label shares in the cluster
                y[i][self.max_labels[cluster]] = self.shares[cluster]
        y = torch.tensor(y, device=self.device)
        return y

    def get_prediction_confidences(self, y):
        return torch.max(y, -1)[0]

    def init_and_fit(self, dataset=None):
        if dataset:
            self.dataset = dataset
        if self.dataset:
            loader = dataset_util.get_loader(self.dataset, shuffle=True)
            self.fit(loader)

    def update_and_fit(self, data, indexes=None, epoch=0):
        if self.dataset is not None:
            for i, index in enumerate(indexes):
                self.dataset.data[index] = data[i].cpu()
            loader = dataset_util.get_loader(self.dataset, shuffle=True)
            self.fit(loader, epoch=epoch)

    def get_threshold(self, normalized=True):
        if normalized:
            return np.quantile(self.valid_shares, self.share_threshold)
        else:
            return self.share_threshold

    def set_threshold(self, threshold):
        if threshold != 'auto':
            self.share_threshold = threshold
        else:
            self.share_threshold = 0.5

    def init_results(self):
        d = dict()
        # d['shares'] = self.valid_shares
        d['shares_mean'] = np.mean(self.valid_shares)
        return d

    def key_param(self):
        return int(self.k / self.n_labels)

    def to_state_dict(self):
        model_dict = dict({
            'centroids': self.model.centroids,
            'type': 'faiss_kmeans',
            'k': self.k,
            'dim': self.dim,
            'niter': self.niter,
            'device': self.device,
            'n_labels': self.n_labels,
            'index': faiss.serialize_index(self.model.index),
            'metadata': {
                'max_labels': self.max_labels,
                'shares': self.shares,
                'valid_shares': self.valid_shares,
                'share_threshold': self.share_threshold,
                'distances_q': self.distances_q,
                'jointly_trained': self.jointly_trained
            }
        })
        return  model_dict


    def from_state_dict(self, model_dict):

        if model_dict['type'] != 'faiss_kmeans':
            raise TypeError("Expected model type 'faiss_kmeans'.")
        self.k = model_dict['k']
        self.dim = model_dict['dim']
        self.niter = model_dict['niter']
        self.device = model_dict['device']
        self.model = faiss.Kmeans(self.dim, self.k, gpu='cuda' in self.device.type)
        self.model.centroids = model_dict['centroids']
        self.model.index = faiss.deserialize_index(model_dict['index'])
        self.max_labels = model_dict['metadata']['max_labels']
        self.shares = model_dict['metadata']['shares']
        self.valid_shares = model_dict['metadata']['valid_shares']
        self.share_threshold = model_dict['metadata']['share_threshold']
        self.distances_q = model_dict['metadata']['distances_q']
        self.jointly_trained = model_dict['metadata']['jointly_trained']

    def save(self, filename):
        model_dict = self.to_state_dict()
        np.save(f"{filename}.npy", model_dict)

    def load(self, filename):
        model_dict = np.load(f"{filename}.npy", allow_pickle=True).item()
        self.from_state_dict(model_dict)

    def eval(self):
        pass

    def to(self, device):
        self.device = device
        self.model.gpu = 'cuda' in device.type
        return self

    def get_cls_loss(self, p, t):
        return torch.nn.modules.loss.CrossEntropyLoss()(p, t)

    def train(self):
        pass
