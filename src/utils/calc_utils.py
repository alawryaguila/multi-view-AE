'''
calc_corr: function for calculating different correlation measurements
ProductOfExperts: class to create product of experts for VAE model
MeanRepresentation: class to create mean representation from separate representations of VAE model
'''
import numpy as np
import torch
import torch.nn as nn
def calc_corr(x1, x2, corr_type='pearson'):
    if corr_type=='pearson':
        return np.corrcoef(x1,x2)

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / var
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

class MeanRepresentation(nn.Module):
    """Return mean of separate VAE representations.
    @param mu: M x D for M views
    @param logvar: M x D for M views
    """
    def forward(self, mu, logvar):
        mean_mu     = torch.mean(mu, axis=0)
        mean_logvar = torch.mean(logvar, axis=0)
        return mean_mu, mean_logvar