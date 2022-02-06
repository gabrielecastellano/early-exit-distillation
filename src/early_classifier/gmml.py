import faiss
import torch
import numpy as np

from structure.logger import MetricLogger
from myutils.pytorch import func_util
from early_classifier.base import BaseClassifier
from early_classifier.gmm_layer.gmml import GMML


class GMMLClassifier(BaseClassifier):

    def __init__(self, device, n_labels, embedding_size, optimizer_config, scheduler_config, criterion_config,
                 n_components=1, batch_size=32, v_batch_size=32, epochs=100, threshold=0.5, components_init=False,
                 cov_type="tril"):
        super().__init__(device, n_labels)
        self.embedding_size = embedding_size
        self.n_components = n_components
        self.cov_type = cov_type
        self.epochs = epochs
        self.model = GMML(embedding_size, round(embedding_size/2), n_labels, cov_type=self.cov_type, n_component=8,
                          log_stretch_trick=False).to(device)
        self.model.sample_parameters()
        self.optimizer = func_util.get_optimizer(self.model, optimizer_config['type'], optimizer_config['params'])
        self.scheduler = func_util.get_scheduler(self.optimizer, scheduler_config['type'], scheduler_config['params'])
        self.criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
        self.batch_size = batch_size
        self.v_batch_size = v_batch_size
        self.threshold = threshold
        self.confidences = []
        self.components_init = components_init

    def _mean_centroids_init(self, dataset):
        print("Initializing components means...")
        targets = torch.tensor(dataset.targets)
        confidences = torch.tensor(dataset.confidences)
        old_t = dataset.transform
        dataset.transform = lambda x: (self.model.bottleneck(torch.tensor(x[0], device=self.device)), x[1])
        mu = torch.zeros(self.n_labels, self.n_components, round(self.embedding_size/2), device=self.device)
        for label in range(self.n_labels):
            z = dataset[(targets == label) & (confidences > 0.99)][0]
            # z = torch.tensor(z)
            _mu = z.mean(0)
            for c in range(self.n_components):
                mu[label][c] = _mu.clone()
        self.model.init_mu(mu)
        dataset.transform = old_t

    def _kmeans_centroids_init(self, dataset):
        print("Initializing components means...")
        old_t = dataset.transform
        dataset.transform = lambda x: (self.model.bottleneck(x[0].to(self.device)), x[1])
        x = np.array(dataset[:][0].cpu())
        y = np.array(dataset.targets)
        k = self.n_components*self.n_labels
        kmeans = faiss.Kmeans(round(self.embedding_size/2), k, niter=10, gpu=False)
        kmeans.train(x)
        centroids = torch.tensor(kmeans.centroids).to(self.device)
        centroid_distances, clusters = kmeans.assign(x)
        label_share = np.zeros([k, self.n_labels], dtype=int)
        max_labels = np.zeros(k, dtype=int)
        cluster_sizes = np.zeros(k, dtype=int)
        shares = np.zeros(k, dtype=float)
        for cluster, target in zip(clusters, y):
            if cluster != -1:
                if target < self.n_labels:
                    label_share[cluster][target] += 1
        for cluster in range(k):
            max_labels[cluster] = np.argmax(label_share[cluster])
            if label_share[cluster][max_labels[cluster]] > 0:
                shares[cluster] = np.max(label_share[cluster]) / np.sum(label_share[cluster])
            cluster_sizes[cluster] = np.sum(label_share[cluster])

        assigned_centroids = []
        clusters = [c for c in range(k)]
        clusters.sort(key=lambda x: shares[x], reverse=True)
        mu = torch.zeros(self.n_labels, self.n_components, round(self.embedding_size / 2), device=self.device)
        omega = torch.zeros(self.n_labels, self.n_components, device=self.device) + 0.01
        for label in range(self.n_labels):
            for c in range(self.n_components):
                for cluster in clusters:
                    if cluster not in assigned_centroids:
                        if max_labels[cluster] == label and cluster_sizes[cluster] > 1:
                            share = shares[cluster]
                            mu[label][c] = centroids[cluster]
                            omega[label][c] = share
                            assigned_centroids.append(cluster)
                            break
        with torch.no_grad():
            self.model.init_mu(mu)
            self.model.init_omega(omega)
            self.model.sample_parameters()
        dataset.transform = old_t

    def fit(self, data_loader, epoch=0):

        metric_logger = MetricLogger(delimiter='  ')
        header = 'TRAIN EE (GMML): epoch {}'.format(epoch)
        self.confidences = []
        self.model.train()
        self.model.mu_p.requires_grad = True
        self.model.sigma_p.requires_grad = True
        if epoch > 30:
            self.model.sigma_p.requires_grad = True
        for i, (sample_batch, targets) in enumerate(metric_logger.log_every(data_loader, round(0.2*len(data_loader.dataset)), header=header)):
            sample_batch, targets = sample_batch.to(self.device), targets.to(self.device)
            if i*sample_batch.shape[0] % self.v_batch_size == 0 or i*sample_batch.shape[0] + sample_batch.shape[0] >= len(data_loader.dataset):
                self.optimizer.zero_grad()
            outputs = self.model.forward(sample_batch)
            loss = self.get_cls_loss(outputs, targets)
            # loss = self.criterion(outputs, targets)
            loss.backward()
            if i * sample_batch.shape[0] % self.v_batch_size == 0 or i*sample_batch.shape[0] + sample_batch.shape[0] >= len(data_loader.dataset):
                self.optimizer.step()
            self.model.sample_parameters()
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])
            self.confidences.extend(outputs.max(dim=-1).values.cpu().detach().numpy())
        self.training_history[epoch] = (metric_logger.lr.value, metric_logger.loss.global_avg)
        self.last_epoch = epoch
        self.scheduler.step()

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            y = self.forward(x)
            # y = torch.softmax(y, dim=-1)
            return y

    def forward(self, x):
        in_device = x.device
        x = x.to(self.device)
        y = torch.zeros(x.size(0), self.n_labels).to(self.device)
        for i in range(0, x.size(0), self.batch_size):
            j = i + self.batch_size
            y[i:j] = self.model.forward(x[i:j])
        return y.to(in_device)

    def get_prediction_confidences(self, y):
        return torch.max(self.get_prediction_probabilities(y), -1)[0]

    def get_threshold(self, normalized=True):
        if normalized:
            return np.quantile(np.array(self.confidences), self.threshold)
        else:
            return self.threshold

    def set_threshold(self, threshold):
        self.threshold = threshold

    def init_results(self):
        d = dict()
        return d

    def key_param(self):
        return self.n_components

    def to_state_dict(self):
        model_dict = dict({
            'type': 'gmml',
            'model': self.model.state_dict(),
            'n_components': self.n_components,
            'embedding_size': self.embedding_size,
            'n_labels': self.n_labels,
            'batch_size': self.batch_size,
            'v_batch_size': self.v_batch_size,
            'threshold': self.threshold,
            'device': self.device,
            'jointly_trained': self.jointly_trained,
            'confidences': self.confidences,
            'last_epoch': self.last_epoch,
            'performance': self.performance
        })
        return model_dict

    def from_state_dict(self, model_dict):
        if model_dict['type'] != 'gmml':
            raise TypeError("Expected model type 'gmml'.")
        self.embedding_size = model_dict['embedding_size']
        self.n_labels = model_dict['n_labels']
        self.n_components = model_dict['n_components']
        self.device = model_dict['device']
        self.model.load_state_dict(model_dict['model'])
        self.model.cov_type = self.cov_type
        self.model.to(self.device)
        self.model.sample_parameters()
        self.batch_size = model_dict['batch_size']
        self.v_batch_size = model_dict['v_batch_size']
        self.threshold = model_dict['threshold']
        self.jointly_trained = model_dict['jointly_trained']
        self.confidences = model_dict['confidences']
        self.last_epoch = model_dict['last_epoch'] if 'last_epoch' in model_dict else -1
        self.performance = model_dict['performance'] if 'performance' in model_dict else 0.0

    def save(self, filename):
        model_dict = self.to_state_dict()
        torch.save(model_dict, open(filename, 'wb'))

    def load(self, filename):
        model_dict = torch.load(filename)
        self.from_state_dict(model_dict)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    def get_model_parameters(self):
        return self.model.parameters()

    def get_cls_loss(self, p, t):
        # return torch.nn.NLLLoss(reduction="sum")(p, t)  # +
        #if self.confidences:
        #    p_ = p[(p > np.quantile(np.array(self.confidences), 0.7)).any(1), :]
        #    t_ = t[(p > np.quantile(np.array(self.confidences), 0.7)).any(1)]
        #else:
        #    p_ = p
        #    t_ = t
        return self.criterion(p, t)

    # Doesn't work for supervised learning. If the max component of any prediction leads to a wrong classification, the
    # total loss is infinite (no matter the other results in the batch).
    @staticmethod
    def _max_component_log_likelihood_loss(p, t):
        return - torch.sum(p.max(dim=-1).values + ((p.max(dim=-1).indices != t).long() * float('-inf')).nan_to_num(nan=0), dim=-1)

    def get_kl_divergence(self, mu, logvar, targets):
        mu = self.model.bottleneck(mu)
        logvar = self.model.bottleneck(logvar)
        var = logvar.exp()
        kld = 0
        for i in range(0, mu.size(0), self.batch_size):
            j = i+self.batch_size
            for c in range(self.n_labels):
                mu_c = mu[i:j][targets[i:j] == c]
                var_c = var[i:j][targets[i:j] == c]
                kld += 0.5 * (self.model.distribution.covariance_matrix[0][c].det().log() - var_c.log().sum(dim=1)
                              - self.model.d
                              + torch.einsum('bs,bs->b',
                                             torch.matmul(mu_c - self.model.distribution.loc[0][c], self.model.distribution.precision_matrix[0][c]),
                                             (mu_c - self.model.distribution.loc[0][c]))
                              + torch.einsum('bii->b', self.model.distribution.precision_matrix[0][c] * var_c.reshape(mu_c.size(0), 1, self.model.d).repeat(1, self.model.d, 1))).sum()
        return kld

    def init_param_from_dataset(self, dataset=None):
        if self.components_init:
            # self._mean_centroids_init(dataset)
            self._kmeans_centroids_init(dataset)
