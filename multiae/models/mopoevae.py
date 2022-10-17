import torch
import hydra
import numpy as np
from ..base.constants import MODEL_MOPOEVAE
from ..base.base_model import BaseModelVAE
from itertools import combinations
from ..base.representations import ProductOfExperts, MixtureOfExperts

class MoPoEVAE(BaseModelVAE):
    """
    Mixture-of-Product-of-Experts Variational Autoencoder: Sutter, Thomas & Daunhawer, Imant & Vogt, Julia. (2021). Generalized Multimodal ELBO. 

    Code is based on: https://github.com/thomassutter/MoPoE 

    Args:
    cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
        model.beta (int, float): KL divergence weighting term.
        encoder._target_ (multiae.models.layers.VariationalEncoder): Type of encoder class to use. 
        encoder.enc_dist._target_ (multiae.base.distributions.Normal, multiae.base.distributions.MultivariateNormal): Encoding distribution.
        decoder._target_ (multiae.models.layers.VariationalDecoder): Type of decoder class to use.
        decoder.init_logvar(int, float): Initial value for log variance of decoder.
        decoder.dec_dist._target_ (multiae.base.distributions.Normal, multiae.base.distributions.MultivariateNormal): Decoding distribution.
        
    input_dim (list): Dimensionality of the input data.
    z_dim (int): Number of latent dimensions. 
    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_MOPOEVAE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)
        self.subsets = self.set_subsets()
    
    def encode(self, x):
        mu = []
        var = []

        for i in range(self.n_views):
            mu_, logvar_ = self.encoders[i](x[i])
            mu.append(mu_) 
            var_ = logvar_.exp()
            var.append(var_) 
        mu = torch.stack(mu)
        var = torch.stack(var)

        mu_out = []
        var_out = []
        if self._training:
            qz_xs = []
            for subset in self.subsets:
                mu_s = mu[subset]
                var_s = var[subset]
                mu_s, var_s = ProductOfExperts()(mu_s, var_s)    
                mu_out.append(mu_s)
                var_out.append(var_s)
                qz_x = hydra.utils.instantiate(
                    self.cfg.encoder.enc_dist, loc=mu_s, scale=var_s.pow(0.5)
                )
                qz_xs.append(qz_x)
            mu_out = torch.stack(mu_out)
            var_out = torch.stack(var_out)
            
            moe_mu, moe_var = MixtureOfExperts()(mu_out, var_out)
            
            qz_x = hydra.utils.instantiate(
                self.cfg.encoder.enc_dist, loc=moe_mu, scale=moe_var.pow(0.5)
                )
            return [qz_xs, qz_x]
        else:
            for subset in self.subsets:
                mu_s = mu[subset]
                var_s = var[subset]
                mu_s, var_s = ProductOfExperts()(mu_s, var_s)    
                mu_out.append(mu_s)
                var_out.append(var_s)
            
            mu_out = torch.stack(mu_out)
            var_out = torch.stack(var_out)

            moe_mu, moe_var = MixtureOfExperts()(mu_out, var_out)
 
            qz_x = hydra.utils.instantiate(
                self.cfg.encoder.enc_dist, loc=moe_mu, scale=moe_var.pow(0.5)
                )
            return [qz_x]

    def decode(self, qz_x):
        px_zs = []
        for i in range(self.n_views):
            px_z = self.decoders[i](qz_x[0]._sample(training=self._training))
            px_zs.append(px_z)
        return [px_zs]

    def forward(self, x):
        qz_xs, qz_x = self.encode(x)
        px_zs = self.decode([qz_x])
        fwd_rtn = {"px_zs": px_zs, "qz_xs_subsets": qz_xs, "qz_x_joint": qz_x}
        return fwd_rtn

    def loss_function(self, x, fwd_rtn):
        px_zs = fwd_rtn["px_zs"]
        qz_xs = fwd_rtn["qz_xs_subsets"]

        kl = self.calc_kl_moe(qz_xs)
        ll = self.calc_ll(x, px_zs)
        total = -ll + self.beta * kl

        losses = {"loss": total, "kl": kl, 'll': ll}
        return losses

    def moe_fusion(self, mus, vars):
        '''
        Implemented from: https://github.com/thomassutter/MoPoE
        '''
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

    def calc_kl_moe(self, qz_xs):
        weight = 1/len(qz_xs)
        kl = 0
        for qz_x in qz_xs:
            kl +=qz_x.kl_divergence(self.prior).sum(1, keepdims=True).mean(0)
            
        return kl*weight

    def set_subsets(self):
        xs = list(range(0, self.n_views))
        
        tmp = [list(combinations(xs, n+1)) for n in range(len(xs))]
        subset_list = [list(item) for sublist in tmp for item in sublist]
        return subset_list
    
    def calc_ll(self, x, px_zs):
        ll = 0
        for i in range(self.n_views):
            ll += px_zs[0][i].log_likelihood(x[i]).sum(1, keepdims=True).mean(0) #first index is latent, second index is view 
        return ll