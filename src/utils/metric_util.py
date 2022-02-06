import time

import torch
import numpy as np

from pytorch_metric_learning import miners, losses, distances, reducers, trainers, samplers, testers
from pytorch_metric_learning.utils import logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from torch import nn
from torch.nn import Module

from early_classifier.ee_dataset import EmbeddingDataset
from structure.logger import MetricLogger, SmoothedValue
from utils import dataset_util


def learn_dataset(dataset, device, n_labels=None, epochs=20):
    """

    Args:
        dataset (EmbeddingDataset):
        device:
        n_labels:
        epochs:

    Returns (Module):

    """
    pin_memory = 'cuda' in device.type
    loader = dataset_util.get_loader(dataset, shuffle=True, n_labels=n_labels, pin_memory=pin_memory)
    sampler = samplers.MPerClassSampler(dataset.targets, m=4, length_before_new_iter=len(dataset))
    record_keeper, _, _ = logging_presets.get_record_keeper("example_logs")
    hooks = logging_presets.get_hook_container(record_keeper)

    input_size = dataset.data.shape[1]
    embedding_size = round(0.5*input_size)
    metric_model = MetricModel([input_size, embedding_size]).to(device)
    trunk = nn.Identity()
    idle = nn.Linear(1, 1)

    #optimizer = torch.optim.SGD(metric_model.parameters(), 0.0001, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(metric_model.parameters(), lr=0.000001, weight_decay=0.0001)
    trunk_optimizer = torch.optim.SGD(idle.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    # loss_fn = losses.CrossBatchMemory(loss=losses.NTXentLoss(temperature=0.1), embedding_size=embedding_size,
    #                                   memory_size=51200)
    # loss_fn = losses.TripletMarginLoss(margin=0.2, distance=distances.CosineSimilarity(), reducer=reducers.ThresholdReducer(low=0))
    loss_fn = losses.TripletMarginLoss(margin=0.1)
    # miner_fn = miners.TripletMarginMiner(margin=0.2, distance=distances.CosineSimilarity(), type_of_triplets="semihard")
    miner_fn = miners.MultiSimilarityMiner(epsilon=0.1)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    tester = testers.GlobalEmbeddingSpaceTester(
        end_of_testing_hook=hooks.end_of_testing_hook,
        dataloader_num_workers=2,
        accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
    )
    dataset_dict = dataset_dict = {"val": dataset}
    model_folder = "example_saved_models"
    end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder)

    trainer = trainers.MetricLossOnly(
        {"trunk": trunk, "embedder": metric_model.to(device)},
        {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": optimizer},
        32,
        {"metric_loss": loss_fn},
        {"tuple_miner": miner_fn},
        dataset,
        lr_schedulers={"embedder_scheduler_by_epoch":scheduler},
        sampler=sampler,
        dataloader_num_workers=2,
        end_of_iteration_hook=hooks.end_of_iteration_hook,
        end_of_epoch_hook=end_of_epoch_hook,
    )

    print("Metric Learning")
    trainer.train(num_epochs=epochs)
    '''
    metric_model.to(device)
    for epoch in range(epochs):
        _train(metric_model, loss_fn, miner_fn, device, loader, optimizer, epoch)
        # test(dataset1, dataset2, model, accuracy_calculator)
        # scheduler.step()
    '''
    return metric_model


def transform_dataset(metric_model, dataset, device):
    """

    Args:
        metric_model (Module):
        dataset (EmbeddingDataset):
        device:

    Returns (EmbeddingDataset):

    """
    print("Transform embeddings")
    embeddings = torch.zeros(len(dataset.data), metric_model.embedding_size)


    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    header = ''

    pin_memory = 'cuda' in device.type
    loader = dataset_util.get_loader(dataset, shuffle=False, n_labels=len(dataset), pin_memory=pin_memory)

    with torch.no_grad():
        metric_model.eval()
        metric_model.to(device)
        for i, (sample_batch, targets) in enumerate(metric_logger.log_every(loader, len(dataset), header)):
            start_time = time.time()
            sample_batch, targets = sample_batch.to(device), targets.to(device)
            embeddings[i*loader.batch_size: i*loader.batch_size + sample_batch.shape[0]] = metric_model(sample_batch).cpu()
            metric_logger.meters['img/s'].update(sample_batch.shape[0] / (time.time() - start_time))

    return EmbeddingDataset(embeddings, dataset.targets, dataset.confidences)

def _train(model, loss_fn, mining_func, device, train_loader, optimizer, epoch):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    #metric_logger.add_meter('n_triplets', SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    model.train()
    for sample_batch, targets in metric_logger.log_every(train_loader, len(train_loader.dataset), header):
        start_time = time.time()
        sample_batch, targets = sample_batch.to(device), targets.to(device)
        optimizer.zero_grad()
        embeddings = model(sample_batch)
        indices_tuple = mining_func(embeddings, targets)
        loss = loss_fn(embeddings, targets, indices_tuple)
        loss.backward()
        optimizer.step()
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(sample_batch.shape[0] / (time.time() - start_time))
        #metric_logger.meters['n_triplets'].update(mining_func.num_triplets)


class MetricModel(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.embedding_size = layer_sizes[-1]
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)