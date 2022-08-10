import torch
import torch.nn as nn

# from scipy import stats

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

#TODO: not used
# def calc_corr(x1, x2, corr_type="pearson"):
#     # TODO - write test
#     if corr_type == "pearson":
#         return np.corrcoef(x1, x2)
#     if corr_type == "kendalltau":
#         return stats.kendalltau(x1, x2)
#     if corr_type == "spearman":
#         return stats.spearmanr(x1, x2)
