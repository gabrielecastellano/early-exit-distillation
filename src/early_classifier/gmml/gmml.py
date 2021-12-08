import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from early_classifier.sgdm.torch_ard import LinearARD
from torch.distributions.multivariate_normal import MultivariateNormal


class GMML(nn.Module):
    def __init__(self, input_dim, d, n_class, n_component=1, cov_type="full", **kwargs):
        """An GMM layer, which can be used as a last layer for a classification neural network.
        Attributes:
            input_dim (int): Input dimension
            n_class (int): The number of classes
            n_component (int): The number of Gaussian components
            cov_type: (str): The type of covariance matrices. If "diag", diagonal matrices are used, which is computationally advantageous. If "full", the model uses full rank matrices that have high expression capability at the cost of increased computational complexity.
        """
        super(GMML, self).__init__(**kwargs)
        assert input_dim > 0
        assert n_class > 1
        assert n_component > 0
        assert cov_type in ["diag", "full"]
        self.input_dim = input_dim
        self.d = input_dim
        self.s = n_class
        self.g = n_component
        self.cov_type = cov_type
        self.n_total_component = n_component*n_class
        # self.ones_mask = (torch.triu(torch.ones(input_dim, input_dim)) == 1)
        # Bias term will be set in the linear layer so we omitted "+1"
        #if cov_type == "diag":
        #    self.H = int(2 * self.input_dim)
        #else:
        #    self.H = int(self.input_dim * (self.input_dim + 3) / 2)
        # Network
        self.bottleneck = nn.Identity() # nn.Linear(self.input_dim, self.d)
        self.mu = Parameter(torch.rand(self.s, self.g, self.d) - 0.5, requires_grad=True)
        self.omega = Parameter(torch.randn(self.s, self.g), requires_grad=True)
        self._last_log_likelihood = None
        if self.cov_type == "full":
            self.sigma_p = Parameter(torch.randn(self.s, self.g, self.d, self.d), requires_grad=True)
            with torch.no_grad():
                for s in range(self.s):
                    for g in range(self.g):
                        self.sigma_p[s][g] = torch.eye(self.d, requires_grad=True)
        else:
            self.sigma_p = Parameter(torch.ones(self.s, self.g, self.d), requires_grad=True)


    def forward(self, x):
        output = torch.zeros(x.shape[0], self.s)
        x = self.bottleneck(x)
        # SIGMA - symmetric positive definite
        sigma_p = self.sigma_p
        if self.cov_type == "diag":
            sigma_p = torch.diag_embed(sigma_p)
        m = torch.matmul(sigma_p.transpose(-2, -1), sigma_p)
        sigma = m + 0.01*torch.diag_embed(torch.linalg.eig(m).eigenvalues.real)*torch.eye(self.d)
        # MU - no transformation
        mu = self.mu
        # OMEGA - should sum up to 1
        om = torch.softmax(self.omega, -1)
        for node in range(self.s):
            distribution = MultivariateNormal(mu[node][0], sigma[node][0])
            for i in range(x.shape[0]):
                output[i][node] = distribution.log_prob(x[i])
        return output
