import torch
import hydra

from ..base.constants import MODEL_MCVAE
from ..base.base_model import BaseModelVAE
from ..base.distributions import Normal

class mcVAE(BaseModelVAE):
    r"""
    Multi-Channel Variational Autoencoder and Sparse Multi-Channel Variational Autoencoder. 

    Code is based on: https://github.com/ggbioing/mcvae

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            model.beta (int, float): KL divergence weighting term.
            model.sparse (bool): Whether to enforce sparsity of the encoding distribution.
            model.threshold (float): Dropout threshold applied to the latent dimensions. Default is 0.
            encoder._target_ (multiae.architectures.mlp.VariationalEncoder): Type of encoder class to use.
            encoder.enc_dist._target_ (multiae.base.distributions.Normal, multiae.base.distributions.MultivariateNormal): Encoding distribution.
            decoder._target_ (multiae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
            decoder.init_logvar(int, float): Initial value for log variance of decoder.
            decoder.dec_dist._target_ (multiae.base.distributions.Normal, multiae.base.distributions.MultivariateNormal): Decoding distribution.
            
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions. 
    
    References
    ----------
    Antelmi, Luigi & Ayache, Nicholas & Robert, Philippe & Lorenzi, Marco. (2019). Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of Heterogeneous Data. 
    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_MCVAE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

    def encode(self, x):
        qz_xs = []
        for i in range(self.n_views):
            mu, logvar = self.encoders[i](x[i])
            qz_x = hydra.utils.instantiate(
                self.cfg.encoder.enc_dist, loc=mu, scale=logvar.exp().pow(0.5)
            )
            qz_xs.append(qz_x)
        return qz_xs

    def decode(self, qz_xs):
        px_zs = []
        for qz_x in qz_xs:
            px_z = [
                self.decoders[i](qz_x._sample(training=self._training))
                for i in range(self.n_views)
            ]
            px_zs.append(px_z)
            del px_z
        return px_zs

    def forward(self, x):
        qz_xs = self.encode(x)
        px_zs = self.decode(qz_xs)
        fwd_rtn = {"px_zs": px_zs, "qz_xs": qz_xs}
        return fwd_rtn

    def loss_function(self, x, fwd_rtn):
        px_zs = fwd_rtn["px_zs"]
        qz_xs = fwd_rtn["qz_xs"]
        kl = self.calc_kl(qz_xs)
        ll = self.calc_ll(x, px_zs)
        total = kl - ll
        losses = {"loss": total, "kl": kl, "ll": ll}
        return losses

    def calc_kl(self, qz_xs):
        kl = 0
        for qz_x in qz_xs:
            if self.sparse:
                kl += qz_x.sparse_kl_divergence().sum(1, keepdims=True).mean(0)
            else:
                kl += qz_x.kl_divergence(self.prior).sum(1, keepdims=True).mean(0)
        return self.beta * kl

    def calc_ll(self, x, px_zs):
        ll = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                ll += px_zs[j][i].log_likelihood(x[i]).sum(1, keepdims=True).mean(0) #first index is latent, second index is view   
        return ll
