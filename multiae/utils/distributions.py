import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal, kl_divergence

class MultivariateNormal(MultivariateNormal):
    def __init__(
        self,
        loc,
        scale,
        *args, 
        **kwargs
        ):
    
        super().__init__(loc, torch.diag_embed(scale))

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
    
    def log_likelihood(self, x):
        return self.log_prob(x)