class BaseClassifier:

    def __init__(self, device, n_labels):
        self.device = device
        self.n_labels = n_labels

    def fit(self, x, y, c):
        """

        Args:
            x (ndarray): array containing the data to fit
            y (ndarray): array containing the labels to fit
            c (ndarray): array of confidence values for the labels in y

        Returns:

        """
        raise NotImplementedError('fit method must be implemented')

    def predict(self, x):
        """

        Args:
            x (ndarray): the batch of data to predict

        Returns (float, int):
            ndarray: array of predicted labels
            ndarray: array of prediction confidences

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

    def save(self, filename):
        raise NotImplementedError('predict method must be implemented')

    def load(self, filename):
        raise NotImplementedError('predict method must be implemented')

    def get_prediction_confidences(self, y):
        raise NotImplementedError('predict method must be implemented')
