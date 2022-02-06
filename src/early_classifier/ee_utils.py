import numpy as np

from typing import Dict, Type

from early_classifier.base import BaseClassifier
from early_classifier.faiss_kmeans import FaissKMeansClassifier
from early_classifier.faiss_knn import FaissKNNClassifier
from early_classifier.gmml import GMMLClassifier
from early_classifier.knn import KNNClassifier
from early_classifier.linear import LinearClassifier
from early_classifier.kmeans import KMeansClassifier
from early_classifier.sdgm import SDGMClassifier

models: Dict[str, Type[BaseClassifier]] = {
    'kmeans': KMeansClassifier,
    'linear': LinearClassifier,
    'faiss_kmeans': FaissKMeansClassifier,
    'sdgm': SDGMClassifier,
    'gmm_layer': GMMLClassifier,
    'knn': KNNClassifier,
    'faiss_knn': FaissKNNClassifier
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
                 'k': clusters_per_class*classes_subset}
                for classes_subset in params['labels_subsets']
                for clusters_per_class in params['clusters_per_labels']], thresholds
    elif ee_type == 'linear':
        return [{'device': device,
                 'n_labels': classes_subset,
                 'embedding_size': np.prod(bn_shape),
                 'optimizer_config': params['optimizer'],
                 'scheduler_config': params['scheduler'],
                 'criterion_config': params['criterion'],
                 'validation_dataset': None,
                 'batch_size': params['batch_size'],
                 'epochs': params['epoch']}
                for classes_subset in params['labels_subsets']], thresholds
    elif ee_type == 'faiss_kmeans':
        return [{'device': device,
                 'n_labels': classes_subset,
                 'k': clusters_per_class*classes_subset,
                 'dim': np.prod(bn_shape),
                 'threshold': thresholds}
                for classes_subset in params['labels_subsets']
                for clusters_per_class in params['clusters_per_labels']], thresholds
    elif ee_type == 'sdgm':
        return [{'device': device,
                 'n_labels': classes_subset,
                 'embedding_size': np.prod(bn_shape),
                 'optimizer_config': params['optimizer'],
                 'scheduler_config': params['scheduler'],
                 'validation_dataset': None,
                 'per_label_components': params['per_label_components'],
                 'batch_size': params['batch_size'],
                 'epochs': params['epoch']}
                for classes_subset in params['labels_subsets']], thresholds
    elif ee_type == 'gmm_layer':
        return [{'device': device,
                 'n_labels': classes_subset,
                 'n_components': params['n_components'],
                 'cov_type': params['cov_type'],
                 'embedding_size': np.prod(bn_shape),
                 'optimizer_config': params['optimizer'],
                 'scheduler_config': params['scheduler'],
                 'criterion_config': params['criterion'],
                 'batch_size': params['batch_size'],
                 'v_batch_size': params['v_batch_size'],
                 'epochs': params['epoch'],
                 'components_init': params['components_init']}
                for classes_subset in params['labels_subsets']], thresholds
    elif ee_type == 'knn':
        return [{'device': device,
                 'n_labels': classes_subset,
                 'k': n_neighbors,
                 'threshold': thresholds}
                for classes_subset in params['labels_subsets']
                for n_neighbors in params['n_neighbors']], thresholds
    elif ee_type == 'faiss_knn':
        return [{'device': device,
                 'n_labels': classes_subset,
                 'k': n_neighbors,
                 'dim': np.prod(bn_shape),
                 'threshold': thresholds}
                for classes_subset in params['labels_subsets']
                for n_neighbors in params['n_neighbors']], thresholds
    else:
        raise UnknownEETypeError(ee_type)


def num_ee_models_variants(ee_config, device, bn_shape):
    ee_params, _ = iterate_configurations(ee_config['type'], ee_config['params'], device, bn_shape, ee_config['thresholds'])
    return len(ee_params)


def get_ee_model(ee_config, device, bn_shape, pre_trained=False, conf_idx=-1):
    ee_params, threshold = iterate_configurations(ee_config['type'], ee_config['params'], device, bn_shape, ee_config['thresholds'])
    ee_params, threshold = ee_params[conf_idx], threshold[-1]
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
        elif ee_config['type'] == 'sdgm':
            filename = ee_config['ckpt'].format(ee_params['n_labels'], ee_config['samples_fraction'], ee_model.key_param())
        elif ee_config['type'] == 'gmm_layer':
            filename = ee_config['ckpt'].format(ee_params['n_labels'], ee_config['samples_fraction'], ee_model.key_param())
        elif ee_config['type'] == 'knn':
            filename = ee_config['ckpt'].format(ee_params['n_labels'], ee_config['samples_fraction'], ee_model.key_param())
        elif ee_config['type'] == 'faiss_knn':
            filename = ee_config['ckpt'].format(ee_params['n_labels'], ee_config['samples_fraction'], ee_model.key_param())
        else:
            raise UnknownEETypeError(ee_config['type'])
        try:
            ee_model.load(filename)
            ee_model.set_threshold(threshold)
        except FileNotFoundError:
            ee_model = None
    return ee_model


def get_model_type(model_cls):
    for k, c in models.items():
        if c == model_cls:
            return k


'''
def requires_normalization(model_type):
    if model_type == 'gmm_layer':
        return True
    else:
        return False
'''
