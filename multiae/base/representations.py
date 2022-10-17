import torch
import torch.nn as nn

class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    var (torch.Tensor): Variance of experts distribution. M x D for M experts
    """

    def forward(self, mu, var, eps=1e-8):
        T = 1.0 / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1.0 / torch.sum(T, dim=0)
        return pd_mu, pd_var

class MixtureOfExperts(nn.Module):
    """Return parameters for mixture of independent experts.
    Implementation from: https://github.com/thomassutter/MoPoE

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    var (torch.Tensor): Variance of experts distribution. M x D for M experts
    """

    def forward(self, mus, vars):
        num_components = mus.shape[0]
        num_samples = mus.shape[1]
        weights = (1/num_components) * torch.ones(num_components)
        idx_start = []
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k-1])
            if k == num_components-1:
                i_end = num_samples
            else:
                i_end = i_start + int(torch.floor(num_samples*weights[k]))
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples

        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])
        var_sel = torch.cat([vars[k, idx_start[k]:idx_end[k], :] for k in range(num_components)])

        return mu_sel, var_sel


class MeanRepresentation(nn.Module):
    """Return mean of separate VAE representations.
    
    Args:
    mu (torch.Tensor): Mean of distributions. M x D for M views.
    var (torch.Tensor): Variance of distributions. M x D for M views.
    """

    def forward(self, mu, var):
        mean_mu = torch.mean(mu, axis=0)
        mean_var = torch.mean(var, axis=0)
        return mean_mu, mean_var
