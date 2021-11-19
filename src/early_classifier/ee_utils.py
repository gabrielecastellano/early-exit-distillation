import numpy as np

from typing import Dict, Type

from early_classifier.base import BaseClassifier
from early_classifier.faiss_kmeans import FaissKMeansClassifier
from early_classifier.linear import LinearClassifier
from early_classifier.kmeans import KMeansClassifier
from early_classifier.sdgm import SDGMClassifier

models: Dict[str, Type[BaseClassifier]] = {
    'kmeans': KMeansClassifier,
    'linear': LinearClassifier,
    'faiss_kmeans': FaissKMeansClassifier,
    'sdgm': SDGMClassifier
}


class UnknownEETypeError(BaseException):
    """Raised when the chosen ee_type is unknown"""
    def __init__(self, ee_type):
        self.message = f'Unknown early exit model type {ee_type}'
        super().__init__(self.message)


def iterate_configurations(ee_type, params, device, bn_shape, thresholds='auto'):
    if type(thresholds) != list:
        thresholds = [thresholds]
    if ee_type == 'kmeans':
        return [{'n_labels': classes_subset,
                 'k': clusters_per_class*classes_subset,
                 'threshold': threshold}
                for classes_subset in params['labels_subsets']
                for clusters_per_class in params['clusters_per_labels']
                for threshold in thresholds]
    elif ee_type == 'linear':
        return [{'device': device,
                 'n_labels': classes_subset,
                 'embedding_size': np.prod(bn_shape),
                 'optimizer_config': params['optimizer'],
                 'scheduler_config': params['scheduler'],
                 'criterion_config': params['criterion'],
                 'validation_dataset': None,
                 'batch_size': params['batch_size'],
                 'epochs': params['epoch'],
                 'threshold': threshold}
                for classes_subset in params['labels_subsets']
                for threshold in thresholds]
    if ee_type == 'faiss_kmeans':
        return [{'device': device,
                 'n_labels': classes_subset,
                 'k': clusters_per_class*classes_subset,
                 'dim': np.prod(bn_shape),
                 'threshold': threshold}
                for classes_subset in params['labels_subsets']
                for clusters_per_class in params['clusters_per_labels']
                for threshold in thresholds]
    elif ee_type == 'sdgm':
        return [{'device': device,
                 'n_labels': classes_subset,
                 'embedding_size': np.prod(bn_shape),
                 'optimizer_config': params['optimizer'],
                 'scheduler_config': params['scheduler'],
                 'validation_dataset': None,
                 'per_label_components': params['per_label_components'],
                 'batch_size': params['batch_size'],
                 'epochs': params['epoch'],
                 'threshold': threshold}
                for classes_subset in params['labels_subsets']
                for threshold in thresholds]
    else:
        raise UnknownEETypeError(ee_type)


def get_ee_model(ee_config, device, pre_trained=False):
    ee_params = iterate_configurations(ee_config['type'], ee_config['params'], device, ee_config['threshold'])[-1]
    ee_model = models[ee_config['type']](**ee_params)
    if pre_trained:
        if ee_config['type'] == 'kmeans':
            filename = ee_config['ckpt'].format(ee_params['n_labels'], ee_config['samples_fraction'], ee_model.key_param())
        elif ee_config['type'] == 'linear':
            filename = ee_config['ckpt'].format(ee_params['n_labels'], ee_config['samples_fraction'], ee_model.key_param())
        elif ee_config['type'] == 'faiss_kmeans':
            filename = ee_config['ckpt'].format(ee_params['n_labels'], ee_config['samples_fraction'], ee_model.key_param())
        elif ee_config['type'] == 'linear':
            filename = ee_config['ckpt'].format(ee_params['n_labels'], ee_config['samples_fraction'], ee_model.key_param())
        else:
            raise UnknownEETypeError(ee_config['type'])
        ee_model.load(filename)
    return ee_model
