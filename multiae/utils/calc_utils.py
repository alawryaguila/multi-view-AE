import numpy as np
import torch
import torch.nn as nn
from scipy import stats

def calc_corr(x1, x2, corr_type="pearson"):
    # TODO - write test
    if corr_type == "pearson":
        return np.corrcoef(x1, x2)
    if corr_type == "kendalltau":
        return stats.kendalltau(x1, x2)
    if corr_type == "spearman":
        return stats.spearmanr(x1, x2)


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    :param mu: M x D for M experts
    :param logvar: M x D for M experts
    """

    def forward(self, mu, var, eps=1e-8):
      #  var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1.0 / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1.0 / torch.sum(T, dim=0)
        return pd_mu, pd_var


class MeanRepresentation(nn.Module):
    """Return mean of separate VAE representations.
    :param mu: M x D for M views
    :param var: M x D for M views
    """

    def forward(self, mu, var):
        mean_mu = torch.mean(mu, axis=0)
        mean_var = torch.mean(var, axis=0)
        return mean_mu, mean_var


def compute_log_alpha(mu, logvar):
    # clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
    return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(min=-8, max=8)


def compute_logvar(mu, log_alpha):
    return log_alpha + 2 * torch.log(torch.abs(mu) + 1e-8)


def compute_mse(x, y):
    return torch.mean(((x - y) ** 2).sum(dim=-1))


def update_dict(orig_dict, update_dict):
    for key, val in update_dict.items():
        if key in orig_dict.keys():
            orig_dict[key] = val
    return orig_dict


def check_batch_size(batch_size, x):
    if batch_size is None:
        return x[0].shape[0] if (type(x) == list or type(x) == tuple) else x.shape[0]
    else:
        return batch_size
