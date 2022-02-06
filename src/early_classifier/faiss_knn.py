import pickle

import numpy as np
import torch
import faiss

from early_classifier.base import BaseClassifier
from early_classifier.ee_dataset import EmbeddingDataset
from utils import dataset_util


class FaissKNNClassifier(BaseClassifier):


    def __init__(self, device, n_labels, k, dim, threshold=0):
        super().__init__(device, n_labels)
        self.requires_full_fit = True
        self.k = k
        self.dim = dim
        self.model = faiss.IndexFlatL2(int(dim))
        if 'cuda' in device.type:
            self.model = faiss.index_cpu_to_all_gpus(self.model)
        if type(threshold) == list:
            self.threshold = threshold[0]
        self.threshold = self.threshold if self.threshold != 'auto' else 0.5
        self.confidences = []
        self.dataset = None
        self.y = None

    def fit(self, data_loader, epoch=0):
        """

        Args:
            data_loader (DataLoader):
            epoch:

        """
        if type(data_loader.dataset) != EmbeddingDataset:
            raise TypeError(f"Unexpected dataset type '{type(data_loader.dataset)}'")

        self.dataset = data_loader.dataset

        x = np.array(data_loader.dataset.data)
        y = np.array(data_loader.dataset.targets)
        c = np.array(data_loader.dataset.confidences)

        # Model
        self.model.add(x)
        self.y = y

        # Predict training dataset for automatic heuristic threshold
        # s = self.model.predict_proba(x)
        d, n = self.model.search(x, self.k)
        d = torch.as_tensor(d)
        n = torch.as_tensor(y[n])
        # torch.count_nonzero((n == i), dim=-1)
        # c = torch.stack([1 / (self.k) * (1/d**25 * (n == i).long()).nan_to_num(nan=0, posinf=0).sum(dim=-1)
        #                  for i in range(self.n_labels)]).transpose(0, 1)
        c = torch.stack([(torch.exp(-0.00001*d) * (n == i).long()).nan_to_num(nan=0, posinf=0).sum(dim=-1) / torch.exp(-0.00001*d).nan_to_num(nan=0, posinf=0).sum(dim=-1)
                         for i in range(self.n_labels)]).transpose(0, 1)
        c, l = c.max(dim=-1)
        #c = c[l != y]
        # self._t_up = c.max()
        self.confidences = c * 0.85

    def predict(self, x):

        d, n = self.model.search(np.array(x), self.k)
        d = torch.as_tensor(d)
        n = torch.as_tensor(self.y[n])
        #y = torch.stack([1 / self.k * (1 / d ** 25 * (n == i).long()).nan_to_num(nan=0).sum(dim=-1)
        #                 for i in range(self.n_labels)]).transpose(0, 1)
        y = torch.stack([(torch.exp(-0.00001 * d) * (n == i).long()).nan_to_num(nan=0, posinf=0).sum(dim=-1) / torch.exp(-0.00001 * d).nan_to_num(nan=0, posinf=0).sum(dim=-1)
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
                self.dataset.data[index] = data[i].cpu()
            loader = dataset_util.get_loader(self.dataset, shuffle=True)
            self.fit(loader, epoch=epoch)

    def get_threshold(self, normalized=True):
        if normalized:
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
        # d['shares'] = self.valid_shares
        return d

    def key_param(self):
        return self.k

    def to_state_dict(self):
        model_dict = dict({
            'type': 'faiss_knn',
            'k': self.k,
            'device': self.device,
            'n_labels': self.n_labels,
            'threshold': self.threshold,
            'jointly_trained': self.jointly_trained,
            'confidences': self.confidences,
            'dim': self.dim,
            'model': faiss.serialize_index(self.model),
            'y': self.y
        })
        return  model_dict


    def from_state_dict(self, model_dict):

        if model_dict['type'] != 'faiss_knn':
            raise TypeError("Expected model type 'faiss_knn'.")
        self.k = model_dict['k']
        self.dim = model_dict['dim']
        self.device = model_dict['device']
        self.model = faiss.deserialize_index(model_dict['model'])
        self.y = model_dict['y']
        self.n_labels = model_dict['n_labels']
        self.threshold = model_dict['threshold']
        self.jointly_trained = model_dict['jointly_trained']
        self.confidences = model_dict['confidences']

    def save(self, filename):
        with open(filename, "wb") as f:
            model_dict = self.to_state_dict()
            pickle.dump(model_dict, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            model_dict = pickle.load(f)
            self.from_state_dict(model_dict)

    def to(self, device):
        if 'cuda' in device.type and 'cpu' in self.device.type:
            self.model = faiss.index_cpu_to_all_gpus(self.model)
        self.device = device
        return self

    def get_cls_loss(self, p, t):
        return torch.nn.modules.loss.CrossEntropyLoss()(p, t)

    def train(self):
        pass
