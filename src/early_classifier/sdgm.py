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
        self.embedding_size = 384
        self.components = n_labels * per_label_components
        self.model = SDGM(self.embedding_size, n_labels, n_component=per_label_components, cov_type="full").to(device)
        self.validation_dataset = validation_dataset
        self.optimizer = func_util.get_optimizer(self.model, optimizer_config['type'], optimizer_config['params'])
        self.criterion = ELBOLoss(self.model, torch.nn.functional.cross_entropy).to(device)
        self.scheduler = func_util.get_scheduler(self.optimizer, scheduler_config['type'], scheduler_config['params'])
        self.batch_size = batch_size
        self.threshold = threshold if threshold != 'auto' else 0.5
        self.min_c, self.max_c = 1, 0

    def fit(self, data_loader, epoch=0):

        downsampler = torch.nn.Upsample(scale_factor=0.5)
        metric_logger = MetricLogger(delimiter='  ')
        header = 'TRAIN EE (SDGM): epoch {}'.format(epoch)
        self.model.train()
        def get_kl_weight(epoch_, max_epoch): return min(1, 1e-9 * epoch_ / max_epoch)
        for sample_batch, targets in metric_logger.log_every(data_loader, len(data_loader.dataset), header=header):
            sample_batch, targets = sample_batch.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            sample_batch = sample_batch.reshape((sample_batch.shape[0], 6, 17, 17))
            sample_batch = downsampler(sample_batch)
            sample_batch = sample_batch.reshape((sample_batch.shape[0], self.embedding_size))

            outputs = self.model.forward(sample_batch)
            kl_weight = get_kl_weight(epoch, max(self.epochs, epoch))
            loss = self.criterion(outputs, targets, 1, kl_weight=kl_weight)
            loss.backward()
            self.optimizer.step()
            metric_logger.update(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])
            outputs = outputs + (outputs.min(dim=-1)[0] * -1).reshape(outputs.shape[-2], 1).expand(outputs.shape)
            c_list = torch.max(torch.nn.functional.normalize(outputs), -1)[0]
            self.min_c = min(c_list.min().item(), self.min_c)
            self.max_c = max(c_list.max().item(), self.max_c)
        self.scheduler.step()

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            y = self.forward(x)
            y = y + (y.min(dim=-1)[0] * -1).reshape(y.shape[-2], 1).expand(y.shape)
            y = torch.nn.functional.normalize(y)
            return y

    def forward(self, x):
        downsampler = torch.nn.Upsample(scale_factor=0.5)
        in_device = x.device
        x = x.to(self.device)
        x = x.reshape((x.shape[0], 6, 17, 17))
        x = downsampler(x)
        x = x.reshape((x.shape[0], self.embedding_size))
        y = self.model.forward(x)
        return y.to(in_device)

    def get_prediction_confidences(self, y):
        return torch.max(torch.nn.functional.normalize(y), -1)[0]

    def get_threshold(self, normalized=True):
        if normalized:
            return self.min_c + self.threshold*(self.max_c - self.min_c)
        else:
            return self.threshold

    def set_threshold(self, threshold):
        self.threshold = threshold

    def init_results(self):
        d = dict()
        return d

    def key_param(self):
        return self.components / self.n_labels

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
            'device': self.device,
            'min_c': self.min_c,
            'max_c': self.max_c,
            'jointly_trained': self.jointly_trained
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
        self.min_c = model_dict['min_c']
        self.max_c = model_dict['max_c']
        self.jointly_trained = model_dict['jointly_trained']

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

    def get_cls_loss(self, p, t):
        self.criterion(p, t)

    def train(self):
        self.model.train()
