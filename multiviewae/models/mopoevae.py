import torch
import hydra
import numpy as np
from ..base.constants import MODEL_MOPOEVAE, EPS
from ..base.base_model import BaseModelVAE
from itertools import combinations
from ..base.representations import ProductOfExperts, MixtureOfExperts

class MoPoEVAE(BaseModelVAE):
    r"""
    Mixture-of-Product-of-Experts Variational Autoencoder.

    Code is based on: https://github.com/thomassutter/MoPoE

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            
            - model.beta (int, float): KL divergence weighting term.
            - encoder.default._target_ (multiviewae.architectures.mlp.VariationalEncoder): Type of encoder class to use.
            - encoder.default.enc_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoding distribution.
            - decoder.default._target_ (multiviewae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
            - decoder.default.init_logvar (int, float): Initial value for log variance of decoder.
            - decoder.default.dec_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoding distribution.
    
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.

    References
    ----------
    Sutter, Thomas & Daunhawer, Imant & Vogt, Julia. (2021). Generalized Multimodal ELBO.
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
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): list containing the MoE joint encoding distribution. If training, the model also returns the encoding distribution for each subset. 
        """
        mu = []
        logvar = []

        for i in range(self.n_views):
            mu_, logvar_ = self.encoders[i](x[i])
            mu.append(mu_) 
            logvar.append(logvar_) 
        mu = torch.stack(mu)
        logvar = torch.stack(logvar)

        mu_out = []
        logvar_out = []
        if self._training:
            qz_xs = []
            for subset in self.subsets:
                mu_s = mu[subset]
                logvar_s = logvar[subset]
                mu_s, logvar_s = ProductOfExperts()(mu_s, logvar_s)    
                mu_out.append(mu_s)
                logvar_out.append(logvar_s)
                qz_x = hydra.utils.instantiate( 
                    eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_s, scale=logvar_s.exp().pow(0.5)+EPS
                )
                qz_xs.append(qz_x)
            mu_out = torch.stack(mu_out)
            logvar_out = torch.stack(logvar_out)
            
            moe_mu, moe_logvar = MixtureOfExperts()(mu_out, logvar_out)
            
            qz_x = hydra.utils.instantiate( 
                self.cfg.encoder.default.enc_dist, loc=moe_mu, scale=moe_logvar.exp().pow(0.5)+EPS
                )
            return [qz_xs, qz_x]
        else:
            for subset in self.subsets:
                mu_s = mu[subset]
                logvar_s = logvar[subset]
                mu_s, logvar_s = ProductOfExperts()(mu_s, logvar_s)    
                mu_out.append(mu_s)
                logvar_out.append(logvar_s)

            mu_out = torch.stack(mu_out)
            logvar_out = torch.stack(logvar_out)

            moe_mu, moe_logvar = MixtureOfExperts()(mu_out, logvar_out)
 
            qz_x = hydra.utils.instantiate( 
                self.cfg.encoder.default.enc_dist, loc=moe_mu, scale=moe_logvar.exp().pow(0.5)
                )
            return [qz_x]

    def decode(self, qz_x):
        r"""Forward pass of joint latent dimensions through decoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a single element indicating the shared latent dimensions. 
            The inner list is a n_view element list with the position in the list indicating the decoder index.
        """    
        px_zs = []
        for i in range(self.n_views):
            px_z = self.decoders[i](qz_x[0]._sample(training=self._training))
            px_zs.append(px_z)
        return [px_zs]

    def forward(self, x):
        r"""Apply encode and decode methods to input data to generate the joint and subset latent dimensions and data reconstructions. 
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.
        """
        qz_xs, qz_x = self.encode(x)
        px_zs = self.decode([qz_x])
        fwd_rtn = {"px_zs": px_zs, "qz_xs_subsets": qz_xs, "qz_x_joint": qz_x}
        return fwd_rtn

    def loss_function(self, x, fwd_rtn):
        r"""Calculate MoPoE VAE loss.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing each element of the MoPoE VAE loss.
        """
        px_zs = fwd_rtn["px_zs"]
        qz_xs = fwd_rtn["qz_xs_subsets"]

        kl = self.calc_kl_moe(qz_xs)
        ll = self.calc_ll(x, px_zs)

        total = self.beta * kl - ll 

        losses = {"loss": total, "kl": kl, 'll': ll}
        return losses

    def calc_kl_moe(self, qz_xs):
        r"""Calculate KL-divergence between the each PoE subset posterior and the prior distribution.

        Args:
            qz_xs (list): list of encoding distributions.

        Returns:
            (torch.Tensor): KL-divergence loss.
        """
        weight = 1/len(qz_xs)
        kl = 0
        for qz_x in qz_xs:
            kl +=qz_x.kl_divergence(self.prior).mean(0).sum()

        return kl*weight

    def set_subsets(self):
        """Create combinations of subsets of views.

        Returns:
            subset_list (list): list of unique combinations of n_views.
        """
        xs = list(range(0, self.n_views))
        tmp = [list(combinations(xs, n+1)) for n in range(len(xs))]
        subset_list = [list(item) for sublist in tmp for item in sublist]
        return subset_list

    def calc_ll(self, x, px_zs):
        r"""Calculate log-likelihood loss.

        Args:
            x (list): list of input data of type torch.Tensor.
            px_zs (list): list of decoding distributions.

        Returns:
            ll (torch.Tensor): Log-likelihood loss.
        """
        ll = 0
        for i in range(self.n_views):
            ll += px_zs[0][i].log_likelihood(x[i]).mean(0).sum() #first index is latent, second index is view
        return ll
