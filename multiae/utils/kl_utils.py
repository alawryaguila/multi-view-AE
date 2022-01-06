import torch
import torch.nn.functional as F


def compute_log_alpha(mu, logvar):
    # clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
	return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(min=-8, max=8)

def compute_logvar(mu, log_alpha):
    return log_alpha + 2 * torch.log(torch.abs(mu) + 1e-8)

def compute_mse(x, y):
    return torch.mean(((x- y)**2).sum(dim=-1))

def compute_kl(mu, logvar):
    return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)

def compute_kl_sparse(mu, logvar):
    log_alpha = compute_log_alpha(mu, logvar)
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    neg_KL = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) - k1
    neg_KL = neg_KL.mean(1, keepdims=True).mean(0)
    return -neg_KL

def compute_ll(x, x_recon, dist='gaussian'):
    if dist=='gaussian':
        return x_recon.log_prob(x).sum(1, keepdims=True).mean(0)
    elif dist=='bernoulli':
        return torch.sum(F.binary_cross_entropy(x_recon, x, reduction='none'), dim=1).mean(0)
    else:
        raise NotImplementedError
