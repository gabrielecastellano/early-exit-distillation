import torch

from early_classifier.base import BaseClassifier
from early_classifier.sgdm.SGDM import SDGM
from early_classifier.sgdm.torch_ard import ELBOLoss
from structure.logger import MetricLogger
from myutils.pytorch import func_util


class SDGMClassifier(BaseClassifier):

    def __init__(self, device, n_labels, embedding_size, optimizer_config, scheduler_config,
                 validation_dataset, per_label_components, batch_size=32, epochs=100, threshold=0.5):
        super().__init__(device, n_labels)
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.components = n_labels * per_label_components
        self.model = SDGM(embedding_size, n_labels, n_component=self.components, cov_type="full").to(device)
        self.validation_dataset = validation_dataset
        self.optimizer = func_util.get_optimizer(self.model, optimizer_config['type'], optimizer_config['params'])
        self.criterion = ELBOLoss(self.model, torch.nn.functional.cross_entropy).to(device)
        self.scheduler = func_util.get_scheduler(self.optimizer, scheduler_config['type'], scheduler_config['params'])
        self.batch_size = batch_size
        self.threshold = threshold

    def fit(self, data_loader, epoch=0):

        metric_logger = MetricLogger(delimiter='  ')
        header = 'TRAIN EE (SDGM): epoch {}'.format(epoch)
        self.model.train()
        def get_kl_weight(epoch, max_epoch): return min(1, 1e-9 * epoch / max_epoch)
        for sample_batch, targets in metric_logger.log_every(data_loader, len(data_loader.dataset), header=header):
            sample_batch, targets = sample_batch.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model.forward(sample_batch)
            kl_weight = get_kl_weight(epoch, max(self.epochs, epoch))
            loss = self.criterion(outputs, targets, 1, kl_weight=kl_weight)
            loss.backward()
            self.optimizer.step()
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])
        self.scheduler.step()

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            in_device = x.device
            x = x.to(self.device)
            y = self.model.forward(x)
            _, predicted = torch.max(y.data, 1)
            y = torch.max(y.data, 1)[0]/torch.sum(y.data, 1)
            return y.to(in_device)

    def get_prediction_confidences(self, y):
        return torch.max(y, -1)[0]

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, threshold):
        self.threshold = threshold

    def init_results(self):
        d = dict()
        return d

    def key_param(self):
        return self.components

    def to_state_dict(self):
        model_dict = dict({
            'type': 'sdgm',
            'model': self.model.state_dict(),
            'epochs': self.epochs,
            'embedding_size': self.embedding_size,
            'n_labels': self.n_labels,
            'components': self.components,
            'batch_size': self.batch_size,
            'threshold': self.threshold,
            'device': self.device
        })
        return model_dict

    def from_state_dict(self, model_dict):
        if model_dict['type'] != 'sdgm':
            raise TypeError("Expected model type 'sdgm'.")
        self.embedding_size = model_dict['embedding_size']
        self.n_labels = model_dict['n_labels']
        self.device = model_dict['device']
        self.model.load_state_dict(model_dict['model'])
        self.n_labels = model_dict['n_labels']
        self.components = model_dict['components']
        self.batch_size = model_dict['batch_size']
        self.threshold = model_dict['threshold']

    def save(self, filename):
        model_dict = self.to_state_dict()
        torch.save(model_dict, open(filename, 'wb'))

    def load(self, filename):
        model_dict = torch.load(filename)
        self.from_state_dict(model_dict)

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self
