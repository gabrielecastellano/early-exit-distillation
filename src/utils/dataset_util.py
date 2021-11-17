import copy
import multiprocessing

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torch.utils.data.sampler import Sampler
from torchvision import transforms

from structure.dataset import AdvRgbImageDataset
from utils import data_util


def get_test_transformer(dataset_name, normalizer, compression_type, compressed_size, org_size):
    normal_list = [transforms.CenterCrop(org_size)] if dataset_name == 'imagenet' else []
    normal_list.append(transforms.ToTensor())
    if normalizer is not None:
        normal_list.append(normalizer)

    normal_transformer = transforms.Compose(normal_list)
    if compression_type is None or compressed_size is None:
        return normal_transformer

    if compression_type == 'base':
        comp_list = [transforms.Resize(compressed_size), transforms.Resize(org_size), transforms.ToTensor()]
        if normalizer is not None:
            comp_list.append(normalizer)
        return transforms.Compose(comp_list)
    return normal_transformer


def get_data_loaders(dataset_config, batch_size=100, compression_type=None, compressed_size=None, normalized=True,
                     rough_size=None, reshape_size=(224, 224), test_batch_size=1, jpeg_quality=0, distributed=False,
                     order_labels=False):
    data_config = dataset_config['data']
    dataset_name = dataset_config['name']
    train_file_path = data_config['train']
    valid_file_path = data_config['valid']
    test_file_path = data_config['test']
    normalizer_config = dataset_config['normalizer']
    mean = normalizer_config['mean']
    std = normalizer_config['std']

    if dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            #transforms.RandomResizedCrop(reshape_size[0]),
            transforms.Resize(rough_size),
            transforms.RandomCrop(reshape_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_val = transforms.Compose([
            transforms.Resize(reshape_size),
            # transforms.CenterCrop(reshape_size[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(rough_size),
            transforms.CenterCrop(reshape_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=os.path.expanduser('~/dataset'), train=True, download=True,
                                                          transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root=os.path.expanduser('~/dataset'), train=False, download=True,
                                                      transform=transform_test)
        valid_dataset = torchvision.datasets.CIFAR100(root=os.path.expanduser('~/dataset'), train=False, download=True,
                                                      transform=transform_test)
        # this is used to train the early exit cache
        ctrain_dataset = torchvision.datasets.CIFAR100(root=os.path.expanduser('~/dataset'), train=True, download=True,
                                                      transform=transform_test)
    else:
        train_dataset = AdvRgbImageDataset(train_file_path, reshape_size)
        normalizer = data_util.build_normalizer(train_dataset.load_all_data() if mean is None or std is None else None,
                                                mean, std) if normalized else None
        train_comp_list = [transforms.Resize(rough_size), transforms.RandomCrop(reshape_size)]\
            if rough_size is not None else list()
        train_comp_list.extend([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        valid_comp_list = [transforms.ToTensor()]
        if normalizer is not None:
            train_comp_list.append(normalizer)
            valid_comp_list.append(normalizer)

        train_transformer = transforms.Compose(train_comp_list)
        valid_transformer = transforms.Compose(valid_comp_list)
        test_transformer = get_test_transformer(dataset_name, normalizer, compression_type, compressed_size, reshape_size)
        train_dataset = AdvRgbImageDataset(train_file_path, reshape_size, train_transformer)
        eval_reshape_size = rough_size if dataset_name == 'imagenet' else reshape_size
        if dataset_name == 'imagenet':
            valid_transformer = test_transformer

        valid_dataset = AdvRgbImageDataset(valid_file_path, eval_reshape_size, valid_transformer)
        test_dataset = AdvRgbImageDataset(test_file_path, eval_reshape_size, test_transformer, jpeg_quality)
        ctrain_dataset = AdvRgbImageDataset(train_file_path, eval_reshape_size, test_transformer, jpeg_quality)

    num_cpus = multiprocessing.cpu_count()
    num_workers = data_config.get('num_workers', 0 if num_cpus == 1 else min(num_cpus, 8))
    num_workers = 0
    pin_memory = torch.cuda.is_available()

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset)
        test_sampler = DistributedSampler(test_dataset)
        ctrain_sampler = DistributedSampler(ctrain_dataset)
    elif order_labels:  # FIXME is this needed?
        train_sampler = PerLabelSampler(train_dataset)
        valid_sampler = PerLabelSampler(valid_dataset)
        test_sampler = PerLabelSampler(test_dataset)
        ctrain_sampler = PerLabelSampler(ctrain_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)
        test_sampler = SequentialSampler(test_dataset)
        ctrain_sampler = SequentialSampler(ctrain_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    if isinstance(test_dataset, AdvRgbImageDataset) and 1 <= test_dataset.jpeg_quality <= 100:
        test_dataset.compute_compression_rate()

    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, sampler=test_sampler,
                             num_workers=num_workers, pin_memory=pin_memory)
    ctrain_loader = DataLoader(ctrain_dataset, batch_size=batch_size, sampler=ctrain_sampler,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader, ctrain_loader


def get_datasets(dataset_config, compression_type=None, compressed_size=None, normalized=True, rough_size=None,
                 reshape_size=(224, 224), jpeg_quality=0):
    data_config = dataset_config['data']
    dataset_name = dataset_config['name']
    train_file_path = data_config['train']
    valid_file_path = data_config['valid']
    test_file_path = data_config['test']
    normalizer_config = dataset_config['normalizer']
    mean = normalizer_config['mean']
    std = normalizer_config['std']

    if dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            #transforms.RandomResizedCrop(reshape_size[0]),
            transforms.Resize(rough_size),
            transforms.RandomCrop(reshape_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_val = transforms.Compose([
            transforms.Resize(reshape_size),
            # transforms.CenterCrop(reshape_size[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(rough_size),
            transforms.CenterCrop(reshape_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=os.path.expanduser('~/dataset'), train=True, download=True,
                                                          transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root=os.path.expanduser('~/dataset'), train=False, download=True,
                                                      transform=transform_val)
        valid_dataset = torchvision.datasets.CIFAR100(root=os.path.expanduser('~/dataset'), train=False, download=True,
                                                      transform=transform_val)
    else:
        train_dataset = AdvRgbImageDataset(train_file_path, reshape_size)
        normalizer = data_util.build_normalizer(train_dataset.load_all_data() if mean is None or std is None else None,
                                                mean, std) if normalized else None
        train_comp_list = [transforms.Resize(rough_size), transforms.RandomCrop(reshape_size)]\
            if rough_size is not None else list()
        train_comp_list.extend([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        valid_comp_list = [transforms.ToTensor()]
        if normalizer is not None:
            train_comp_list.append(normalizer)
            valid_comp_list.append(normalizer)

        train_transformer = transforms.Compose(train_comp_list)
        valid_transformer = transforms.Compose(valid_comp_list)
        test_transformer = get_test_transformer(dataset_name, normalizer, compression_type, compressed_size, reshape_size)
        train_dataset = AdvRgbImageDataset(train_file_path, reshape_size, train_transformer)
        eval_reshape_size = rough_size if dataset_name == 'imagenet' else reshape_size
        if dataset_name == 'imagenet':
            valid_transformer = test_transformer

        valid_dataset = AdvRgbImageDataset(valid_file_path, eval_reshape_size, valid_transformer)
        test_dataset = AdvRgbImageDataset(test_file_path, eval_reshape_size, test_transformer, jpeg_quality)

    return train_dataset, valid_dataset, test_dataset


def get_loader(dataset, shuffle=False, order_labels=False, n_labels=None, batch_size=64, pin_memory=False):
    """

    Args:
        dataset (Dataset):
        shuffle (Bool):
        order_labels: 
        n_labels (int):
        batch_size (int):
        pin_memory (Bool):

    Returns (DataLoader):

    """
    sub_dataset = copy.copy(dataset)
    if n_labels is not None:
        dataset.targets = np.array(dataset.targets)
        sub_dataset.targets = np.array(sub_dataset.targets)
        sub_dataset.targets = sub_dataset.targets[dataset.targets < n_labels]
        sub_dataset.data = sub_dataset.data[dataset.targets < n_labels]

    if order_labels:
        sampler = PerLabelSampler(sub_dataset, shuffle=shuffle)
    elif shuffle:
        sampler = RandomSampler(sub_dataset)
    else:
        sampler = SequentialSampler(sub_dataset)
    return DataLoader(sub_dataset, batch_size=batch_size, sampler=sampler, pin_memory=pin_memory)


class PerLabelSampler(Sampler):
    r"""Samples elements ordered per label, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle
        l = [(c, i) for i, c in enumerate(self.data_source.targets)]
        l.sort(key=lambda e: (e[0], random.random()) if shuffle else (e[0], e[1]))
        self.ordered_indexes = [i for _, i in l]

    def __iter__(self):
        if self.shuffle:
            l = [(c, i) for i, c in enumerate(self.data_source.targets)]
            l.sort(key=lambda e: (e[0], random.random()))
            self.ordered_indexes = [i for _, i in l]
        return iter(self.ordered_indexes)

    def __len__(self):
        return len(self.data_source)