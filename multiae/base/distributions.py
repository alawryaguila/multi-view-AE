import torch

from torch.distributions import Normal, kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.utils import broadcast_all
from torch.nn.functional import binary_cross_entropy


def compute_log_alpha(mu, logvar):
    # clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
    return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(min=-8, max=8)

# TODO: test different distributions
class MultivariateNormal(MultivariateNormal):
    def __init__(
            self,
            **kwargs
        ):

        self.loc = kwargs['loc']
        self.scale = kwargs['scale']
        if isinstance(self.scale, int): # TODO: dont like this, check loc is always list?
            self.covariance_matrix = torch.diag_embed(torch.ones(len(self.loc)))
        else:
            self.covariance_matrix = torch.diag_embed(self.scale)
        super().__init__(loc=self.loc, covariance_matrix=self.covariance_matrix)

    @property
    def variance(self):
        return self.scale.pow(2)

    def kl_divergence(self, other):
        kl = kl_divergence(torch.distributions.multivariate_normal.MultivariateNormal( \
                        loc=self.loc, covariance_matrix=self.covariance_matrix), other)
        sh = kl.shape
        return kl.reshape((sh[0], 1))   # TODO: hack. kl_dirgence returns (x,) vector here. not the same with Normal.kl_divergence

    def sparse_kl_divergence(self):  # TODO: check this is also the case for multivariate gauss - fine except using Normal prior
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
        sh = ll.shape
        return ll.reshape((sh[0], 1))

    def _sample(self, training=False):
        if training:
            return self.rsample()
        return self.loc


class Normal(Normal):
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

    def _sample(self, training=False):
        if training:
            return self.rsample()
        return self.loc

# TODO: test this
class Bernoulli():
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
        return torch.distributions.bernoulli.Bernoulli(torch.sigmoid(self.x)).sample()


class Default():
    def __init__(
        self,
        **kwargs,
    ):
        self.x = kwargs['x']


    def log_likelihood(self, x):
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
