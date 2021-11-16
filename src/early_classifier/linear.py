import torch

from torch.utils.data import Dataset, RandomSampler

from early_classifier.base import BaseClassifier
from early_classifier.ee_dataset import CustomDataset
from structure.logger import MetricLogger
from myutils.pytorch import func_util


class LinearClassifier(BaseClassifier):


    def __init__(self, device, n_labels, embedding_size, optimizer_config, scheduler_config, criterion_config,
                 validation_dataset, batch_size=32, epochs=100, threshold=0.5):
        super().__init__(device, n_labels)
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.model = torch.nn.Linear(embedding_size, n_labels).to(device)
        self.validation_dataset = validation_dataset
        self.optimizer = func_util.get_optimizer(self.model, optimizer_config['type'], optimizer_config['params'])
        self.criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
        self.scheduler = func_util.get_scheduler(self.optimizer, scheduler_config['type'], scheduler_config['params'])
        self.batch_size = batch_size
        self.threshold = threshold

    def fit(self, x, y, c, epoch=0):

        train_dataset = CustomDataset(x, y, n_classes=self.n_labels)
        # train_sampler = PerLabelSampler(train_dataset, shuffle=True)
        train_sampler = RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler)

        metric_logger = MetricLogger(delimiter='  ')
        header = 'TRAIN EE (LINEAR): epoch {}'.format(epoch)
        self.model.train()
        for sample_batch, targets in metric_logger.log_every(train_loader, 100, header=header):
            sample_batch, targets = sample_batch.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model.forward(sample_batch)
            loss = self.criterion(outputs, targets)
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
            y = torch.softmax(y, dim=-1)
            return y.to(in_device)

    def get_prediction_confidences(self, y):
        return torch.max(y, -1)[0]

    def get_threshold(self):
        return self.threshold

    def init_results(self):
        d = dict()
        return d

    def key_param(self):
        return self.get_threshold()

    def to_state_dict(self):
        model_dict = dict({
            'type': 'linear',
            'model': self.model.state_dict(),
            'epochs': self.epochs,
            'embedding_size': self.embedding_size,
            'n_labels': self.n_labels,
            'batch_size': self.batch_size,
            'threshold': self.threshold,
            'device': self.device
        })
        return model_dict

    def from_state_dict(self, model_dict):
        if model_dict['type'] != 'linear':
            raise TypeError("Expected model type 'linear'.")
        self.embedding_size = model_dict['embedding_size']
        self.n_labels = model_dict['n_labels']
        self.device = model_dict['device']
        self.model = torch.nn.Linear(self.embedding_size, self.n_labels).to(self.device)
        self.n_labels = model_dict['n_labels']
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
