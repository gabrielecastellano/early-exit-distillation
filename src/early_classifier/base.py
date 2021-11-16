class BaseClassifier:

    def __init__(self, device, n_labels):
        self.device = device
        self.n_labels = n_labels

    def fit(self, x, y, c, epoch=0):
        """

        Args:
            x (Tensor): array containing the data to fit
            y (Tensor): array containing the labels to fit
            c (Tensor): array of confidence values for the labels in y
            epoch:

        Returns:

        """
        raise NotImplementedError('fit method must be implemented')

    def predict(self, x):
        """

        Args:
            x (Tensor): the batch of data to predict

        Returns (float, int):
            Tensor: array of predicted labels
            Tensor: array of prediction confidences

        """
        raise NotImplementedError('predict method must be implemented')

    def get_threshold(self):
        """

        Returns: Basic classification threshold value

        """
        raise NotImplementedError('predict method must be implemented')

    def init_results(self):
        return dict()

    def key_param(self):
        raise NotImplementedError('predict method must be implemented')

    def to_state_dict(self):
        raise NotImplementedError('predict method must be implemented')

    def from_state_dict(self, model_dict):
        raise NotImplementedError('predict method must be implemented')

    def save(self, filename):
        raise NotImplementedError('predict method must be implemented')

    def load(self, filename):
        raise NotImplementedError('predict method must be implemented')

    def get_prediction_confidences(self, y):
        raise NotImplementedError('predict method must be implemented')

    def eval(self):
        raise NotImplementedError('predict method must be implemented')
