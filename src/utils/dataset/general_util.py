import os

import torch.utils.data as data
import torchvision.transforms as transforms

from autoencoders import *
from structure.dataset import AdvRgbImageDataset
from utils import data_util


def get_test_transformer(normalizer, compression_type, compressed_size, org_size, ae_model=None):
    normal_list = [transforms.ToTensor()]
    if ae_model is not None:
        normal_list.append(AETransformer(ae_model))

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


def get_data_loaders(root_data_dir_path, batch_size=100, compression_type=None, compressed_size=None, normalized=True,
                     ae_model=None, rough_size=(256, 256), reshape_size=(224, 224), compression_quality=0):
    if not os.path.exists(root_data_dir_path):
        ValueError('Could not find dataset at {}'.format(root_data_dir_path))

    train_file_path = os.path.join(root_data_dir_path, 'train.txt')
    valid_file_path = os.path.join(root_data_dir_path, 'valid.txt')
    test_file_path = os.path.join(root_data_dir_path, 'test.txt')
    train_dataset = AdvRgbImageDataset(train_file_path, reshape_size)
    normalizer = data_util.build_normalizer(train_dataset.load_all_data()) if normalized else None
    train_comp_list = [transforms.Resize(rough_size), transforms.RandomCrop(reshape_size)] if rough_size > 0 else list()
    train_comp_list.extend([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    valid_comp_list = [transforms.ToTensor()]
    if normalizer is not None:
        train_comp_list.append(normalizer)
        valid_comp_list.append(normalizer)

    pin_memory = torch.cuda.is_available()
    train_transformer = transforms.Compose(train_comp_list)
    valid_transformer = transforms.Compose(valid_comp_list)
    train_dataset = AdvRgbImageDataset(train_file_path, reshape_size, train_transformer)
    valid_dataset = AdvRgbImageDataset(valid_file_path, reshape_size, valid_transformer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=pin_memory)
    test_transformer = get_test_transformer(normalizer, compression_type, compressed_size, reshape_size, ae_model)
    test_dataset = AdvRgbImageDataset(test_file_path, reshape_size, test_transformer, compression_quality)
    if 1 <= test_dataset.jpeg_quality <= 95:
        test_dataset.compute_compression_rate()

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True,
                                              num_workers=2, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader
