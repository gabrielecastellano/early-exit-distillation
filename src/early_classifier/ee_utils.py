from typing import Dict, Type

from early_classifier.base import BaseClassifier
from early_classifier.kmeans import KMeansClassifier

models: Dict[str, Type[BaseClassifier]] = {
    'kmeans': KMeansClassifier,
    'fc-softmax': None,
}


class UnknownEETypeError(BaseException):
    """Raised when the chosen ee_type is unknown"""
    def __init__(self, ee_type):
        self.message = f'Unknown early exit model type {ee_type}'
        super().__init__(self.message)


def iterate_configurations(ee_type, params, device):
    if ee_type == 'kmeans':
        return [{'device': device, 'n_labels': classes_subset, 'k': clusters_per_class*classes_subset}
                for classes_subset in params['labels_subsets']
                for clusters_per_class in params['clusters_per_labels']]
    else:
        raise UnknownEETypeError(ee_type)


def get_ee_model(ee_config, device, pre_trained=False):
    ee_params = iterate_configurations(ee_config['type'], ee_config['params'])[-1]
    ee_model = models[ee_config['type']](**ee_params)
    if pre_trained:
        if ee_config['type'] == 'kmeans':
            filename = ee_config['ckpt'].format(ee_params[0], ee_config['used_samples_per_class'], ee_model.key_param())
        else:
            raise UnknownEETypeError(ee_config['type'])
        ee_model.load(filename)
    return ee_model
