import torch

from torch.distributions import Normal, kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.utils import broadcast_all
from torch.nn.functional import binary_cross_entropy_with_logits

def compute_log_alpha(mu, logvar):
    # clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
    return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(min=-8, max=8)

# TODO: test different options
class MultivariateNormal(MultivariateNormal):
    def __init__(self, loc, scale, *args, **kwargs):

        super().__init__(loc, torch.diag_embed(scale))

    @property
    def variance(self):
        return self.scale.pow(2)

    def kl_divergence(self, other):
        return kl_divergence(MultivariateNormal(loc=self.loc, scale=self.stddev), other)

    def sparse_kl_divergence(
        self,
    ):  # check this is also the case for multivariate gauss
        mu = self.loc
        logvar = torch.log(self.variance)
        log_alpha = compute_log_alpha(mu, logvar)
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        neg_KL = (
            k1 * torch.sigmoid(k2 + k3 * log_alpha)
            - 0.5 * torch.log1p(torch.exp(-log_alpha))
            - k1
        )
        neg_KL = neg_KL.mean(1, keepdims=True).mean(0)
        return -neg_KL

    def log_likelihood(self, x):
        return self.log_prob(x)

    def _sample(self, training=False):
        if training:
            return self.rsample()
        return self.loc


class Normal(Normal):
    def __init__(
        self,
        loc,
        scale,
        *args,
        **kwargs,
    ):
        super().__init__(loc, scale)

    @property
    def variance(self):
        return self.scale.pow(2)

    def kl_divergence(self, other): # TODO: check if scale = stddev
        x = kl_divergence(Normal(loc=self.loc, scale=self.stddev), other)
        return kl_divergence(Normal(loc=self.loc, scale=self.stddev), other)

    # TODO: does not return same shape as kl_divergence
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
        neg_KL = neg_KL.mean(1, keepdims=True).mean(0)
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
        x,
        *args,
        **kwargs,
    ):
        self.x = x

    def log_likelihood(self, x):
        logits, x = broadcast_all(self.x, x)
        return -binary_cross_entropy_with_logits(logits, x, reduction='none')

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
        x,
        *args,
        **kwargs,
    ):
        self.x = x


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
