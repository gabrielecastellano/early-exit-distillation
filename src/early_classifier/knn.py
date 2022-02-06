import sys

import joblib
import numpy as np
import torch
import faiss
from sklearn.neighbors import KNeighborsClassifier

from early_classifier.base import BaseClassifier
from early_classifier.ee_dataset import EmbeddingDataset
from utils import dataset_util


class KNNClassifier(BaseClassifier):

    def __init__(self, device, n_labels, k, threshold=0):
        super().__init__(torch.device('cpu'), n_labels)
        self.requires_full_fit = True
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.label_share = np.zeros([self.k, n_labels], dtype=int)
        self.confidence_share = {cluster: {label: [] for label in range(n_labels)} for cluster in range(self.k)}
        self.max_labels = np.zeros(self.k, dtype=int)
        self.shares = np.zeros(self.k, dtype=float)
        self.cluster_sizes = np.zeros(self.k, dtype=int)
        self.valid_shares = None
        self.threshold = threshold
        if type(threshold) == list:
            self.threshold = threshold[0]
        self.threshold = self.threshold if self.threshold != 'auto' else 0.5
        self._t_up = 1
        self._t_down = 0
        self.confidences = torch.tensor([])
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

        x = torch.as_tensor(data_loader.dataset.data, device=self.device)
        y = torch.as_tensor(data_loader.dataset.targets, device=self.device)
        c = torch.as_tensor(data_loader.dataset.confidences, device=self.device)

        # Model
        self.model.fit(x, y)

        # Predict training dataset for automatic heuristic threshold
        # s = self.model.predict_proba(x)
        d, n = self.model.kneighbors(x)
        n = torch.as_tensor(n)
        d = torch.as_tensor(d)
        n = y[n]
        # torch.count_nonzero((n == i), dim=-1)
        c = torch.stack([1 / (self.k) * (1/d**25 * (n == i).long()).nan_to_num(nan=0, posinf=0).sum(dim=-1)
                         for i in range(self.n_labels)]).transpose(0, 1)
        c, l = c.max(dim=-1)
        #c = c[l != y]
        # self._t_up = c.max()
        self.confidences = c * 0.95

    def predict(self, x):
        # y = self.model.predict_proba(torch.as_tensor(x))
        # y = torch.as_tensor(y)
        d, n = self.model.kneighbors(x)
        n = torch.as_tensor(self.model._y[n])
        d = torch.as_tensor(d)
        y = torch.stack([1 / self.k * (1/d**25 * (n == i).long()).nan_to_num(nan=0).sum(dim=-1)
                         for i in range(self.n_labels)]).transpose(0, 1)
        y = y.to(self.device)
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
                self.dataset.data[index] = data[i].to(self.device)
            loader = dataset_util.get_loader(self.dataset, shuffle=True)
            self.fit(loader, epoch=epoch)

    def get_threshold(self, normalized=True):
        if normalized:
            # return self.confidences[-1]*self.threshold
            return np.quantile(self.confidences, self.threshold)
        else:
            return self.threshold

    def set_threshold(self, threshold):
        if threshold != 'auto':
            self.threshold = threshold
        else:
            self.threshold = 0.5

    def init_results(self):
        d = dict()
        return d

    def key_param(self):
        return self.k

    def to_state_dict(self):
        model_dict = dict({
            'type': 'knn',
            'model': self.model,
            'k': self.k,
            'device': self.device,
            'n_labels': self.n_labels,
            'threshold': self.threshold,
            'jointly_trained': self.jointly_trained,
            'confidences': self.confidences
        })
        return  model_dict


    def from_state_dict(self, model_dict):

        if model_dict['type'] != 'knn':
            raise TypeError("Expected model type 'knn'.")
        self.k = model_dict['k']
        self.device = model_dict['device']
        self.model = model_dict['model']
        self.n_labels = model_dict['n_labels']
        self.threshold = model_dict['threshold']
        self.jointly_trained = model_dict['jointly_trained']
        self.confidences = model_dict['confidences']

    def save(self, filename):
        model_dict = self.to_state_dict()
        joblib.dump(model_dict, open(filename, 'wb'))

    def load(self, filename):
        model_dict = joblib.load(filename)
        self.from_state_dict(model_dict)

    def to(self, device):
        return self

    def get_cls_loss(self, p, t):
        return torch.nn.modules.loss.CrossEntropyLoss()(p, t)

    def train(self):
        pass
