import torch
from torch.distributions import Normal, kl_divergence, Laplace
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.utils import broadcast_all
from torch.nn.functional import binary_cross_entropy
import numpy as np

def compute_log_alpha(mu, logvar):
    # clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
    return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(min=-8, max=8)

class Default():
    """Artificial distribution designed for data with unspecified distribution.
    Used so that log_likelihood and _sample methods can be called by model class.
    Args:
        x (list): List of input data.
    """
    def __init__(
        self,
        **kwargs,
    ):
        self.x = kwargs['x']


    def log_likelihood(self, x):
        """calculates the mean squared error between input data and reconstruction.

        Args:
            x (torch.Tensor): data reconstruction.

        Returns:
            torch.Tensor: Mean squared error.
        """
        logits, x = broadcast_all(self.x, x)
        return - (logits - x)**2

    def rsample(self):
        raise NotImplementedError

    def kl_divergence(self):
        raise NotImplementedError

    def sparse_kl_divergence(self):
        raise NotImplementedError

    def _sample(self):
        return self.x


class Normal(Normal):
    """Univariate normal distribution. Inherits from torch.distributions.Normal.

    Args:
        loc (int, torch.Tensor): Mean of distribution.
        scale (int, torch.Tensor): Standard deviation of distribution.
    """
    def __init__(
        self,
        **kwargs,
    ):
        self.loc = kwargs['loc']
        self.scale = kwargs['scale']
        super().__init__(loc=self.loc, scale=self.scale)

    @property
    def variance(self):
        return self.scale.pow(2)

    def kl_divergence(self, other):
        return kl_divergence(torch.distributions.normal.Normal(loc=self.loc, scale=self.stddev), other)

    def sparse_kl_divergence(self):
        """
        Implementation from: https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb

        """
        mu = self.loc
        logvar = torch.log(self.variance)
        log_alpha = compute_log_alpha(mu, logvar)
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        neg_KL = (
            k1 * torch.sigmoid(k2 + k3 * log_alpha)
            - 0.5 * torch.log1p(torch.exp(-log_alpha))
            - k1
        )
        return -neg_KL

    def log_likelihood(self, x):
        return self.log_prob(x)

    def _sample(self, *kwargs, training=False):
        if training:
            return self.rsample(*kwargs)
        return self.loc


class MultivariateNormal(MultivariateNormal):
    """Multivariate normal distribution with diagonal covariance matrix. Inherits from torch.distributions.multivariate_normal.MultivariateNormal.

    Args:
        loc (list, torch.Tensor): Mean of distribution.
        scale (int, torch.Tensor): Standard deviation of distribution.
    """
    def __init__(
            self,
            **kwargs
        ):

        self.loc = torch.as_tensor(kwargs['loc'])
        self.scale = torch.as_tensor(kwargs['scale'])
        
        #used when fitting encoder/decoder distribution or prior distribution with different mean and SD values
        self.covariance_matrix = torch.diag_embed(self.scale)

        super().__init__(loc=self.loc, covariance_matrix=self.covariance_matrix)

    @property
    def variance(self):
        return self.scale.pow(2)

    def kl_divergence(self, other):

        kl = kl_divergence(torch.distributions.multivariate_normal.MultivariateNormal( \
                        loc=self.loc, covariance_matrix=self.covariance_matrix), other)
        return torch.unsqueeze(kl,-1)

    def sparse_kl_divergence(self):
        mu = self.loc
        logvar = torch.log(self.variance)
        log_alpha = compute_log_alpha(mu, logvar)
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        neg_KL = (
            k1 * torch.sigmoid(k2 + k3 * log_alpha)
            - 0.5 * torch.log1p(torch.exp(-log_alpha))
            - k1
        )
        return -neg_KL

    def log_likelihood(self, x):
        ll = self.log_prob(x)
        return torch.unsqueeze(ll,-1)

    def _sample(self, *kwargs, training=False):
        if training:
            return self.rsample(*kwargs)
        return self.loc


class Bernoulli():
    """Artificial distribution designed for (approximately) Bernoulli distributed data. 
    The data isn't restricted to bernoulli distribution, this class is designed as a wrapper for the log_likelihood() method which is required for the multiview models.

    Args:
        x (list): List of input data.
    """
    def __init__(
        self,
        **kwargs,
    ):
        self.x = kwargs['x']

    def log_likelihood(self, target):   
        x, target = broadcast_all(self.x, target)
        x = torch.sigmoid(x)
        bce = binary_cross_entropy(x, target, reduction='none')
        return -bce

    def rsample(self):
        raise NotImplementedError

    def kl_divergence(self):
        raise NotImplementedError

    def sparse_kl_divergence(self):
        raise NotImplementedError

    def _sample(self):
        return torch.sigmoid(self.x)

class Laplace(Laplace):
    """Laplace distribution. Inherits from torch.distributions.Laplace.

    Args:
        loc (list, torch.Tensor): Mean of distribution.
        scale (int, torch.Tensor): Standard deviation of distribution.
    """
    def __init__(
            self,
            **kwargs
        ):
        if 'loc' in kwargs:
            self.loc = kwargs['loc']
        elif 'x' in kwargs:
            self.loc = kwargs['x']

        if 'scale' in kwargs:
            self.scale = kwargs['scale']
        else: 
            self.scale = torch.tensor(0.75) * torch.ones(self.loc.shape)
            self.scale = self.scale.to(self.mu.device)
        super().__init__(loc=self.loc, scale=self.scale)

    def kl_divergence(self):
        raise NotImplementedError

    def sparse_kl_divergence(self):
        raise NotImplementedError
        
    def log_likelihood(self, x):
        return self.log_prob(x)

    def _sample(self, *kwargs, training=False):
        if training:
            return self.rsample(*kwargs)
        return self.loc

