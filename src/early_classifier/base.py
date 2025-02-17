class BaseClassifier:

    def __init__(self, device, n_labels):
        self.model = None
        self.device = device
        self.n_labels = n_labels
        self.jointly_trained = False
        self.requires_full_fit = False
        self.last_epoch = -1
        self.training_history = dict()
        self.performance = 0.0

    def fit(self, data_loader, epoch=0):
        """

        Args:
            data_loader (torch.utils.data.dataloader.DataLoader): data to fit
            epoch:

        Returns:

        """
        raise NotImplementedError('fit method must be implemented')

    def predict(self, x):
        """
        To be used just for evaluation.
        Args:
            x (Tensor): the batch of data to predict

        Returns:
            Tensor: array of predicted confidences per label

        """
        raise NotImplementedError('predict method must be implemented')

    def forward(self, x):
        """
        Used in training, to be overridden only if should be different from the evaluation prediction.
        Args:
            x (Tensor): the batch of data to predict

        Returns:
            Tensor: array of predicted confidences per label

        """
        return self.predict(x)

    def get_threshold(self, normalized=True):
        """

        Returns: Basic classification threshold value

        """
        raise NotImplementedError('method must be implemented')

    def init_and_fit(self, dataset=None):
        """
        Should set and fit a new dataset only if the model is not optimized through a criterion.backward function.
        Args:
            dataset (EmbeddingDataset):

        Returns:

        """
        pass

    def update_and_fit(self, data, indexes=None, epoch=0):
        """
        Changes a portion of the dataset and performs a new fit (should not do anything in case the model is optimized
        through a criterion.backward function).
        Args:
            data:
            indexes:
            epoch:

        Returns:

        """
        pass

    def init_param_from_dataset(self, dataset=None):
        """
        Initialize model parameters based on the particular dataset, if any parameter can benefit from that.
        Args:
            dataset (EmbeddingDataset):

        Returns:

        """

    def init_results(self):
        return dict()

    def key_param(self):
        raise NotImplementedError('method must be implemented')

    def to_state_dict(self):
        raise NotImplementedError('method must be implemented')

    def from_state_dict(self, model_dict):
        raise NotImplementedError('method must be implemented')

    def save(self, filename):
        raise NotImplementedError('method must be implemented')

    def load(self, filename):
        raise NotImplementedError('method must be implemented')

    def get_prediction_confidences(self, y):
        raise NotImplementedError('method must be implemented')

    def get_prediction_probabilities(self, y):
        return y

    def eval(self):
        pass

    def train(self):
        pass

    def to(self, device):
        raise NotImplementedError('method must be implemented')

    def get_model_parameters(self):
        return list()

    def get_cls_loss(self, p, t):
        raise NotImplementedError('method must be implemented')

    def set_threshold(self, threshold):
        raise NotImplementedError('method must be implemented')
