import torch
import torch.nn.functional as F
from scipy import stats
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal, kl_divergence
from .calc_utils import compute_log_alpha

class MultivariateNormal(MultivariateNormal):
    def __init__(
        self,
        loc,
        scale,
        *args, 
        **kwargs
        ):
    
        super().__init__(loc, torch.diag_embed(scale))

    @property
    def variance(self):
        return self.scale.pow(2)

    def kl_divergence(self, other):
        return kl_divergence(MultivariateNormal(loc=self.loc, scale=self.stddev), other)

    def sparse_kl_divergence(self): #check this is also the case for multivariate gauss
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
        *args, **kwargs,
        ):
        super().__init__(loc, scale)

    @property
    def variance(self):
        return self.scale.pow(2)

    def kl_divergence(self, other):
        return kl_divergence(Normal(loc=self.loc, scale=self.stddev), other)
    
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

class Bernoulli():
    def __init__(
        self,
        x,
        *args, **kwargs,
        ):
        self.x = x

    def log_likelihood(self, x):
        return F.binary_cross_entropy(torch.sigmoid(self.x), x, reduction="none") #check right way around
    
    def rsample(self):
        raise NotImplementedError
    
    def kl_divergence(self):
        raise NotImplementedError

    def sparse_kl_divergence(self):
        raise NotImplementedError
    
    def _sample(self):
        return torch.distributions.bernoulli.Bernoulli(torch.sigmoid(self.x)).sample()
