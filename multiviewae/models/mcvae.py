import torch
import hydra

from ..base.constants import MODEL_MCVAE, EPS
from ..base.base_model import BaseModelVAE
from ..base.distributions import Normal

class mcVAE(BaseModelVAE):
    r"""
    Multi-Channel Variational Autoencoder and Sparse Multi-Channel Variational Autoencoder.

    Code is based on: https://github.com/ggbioing/mcvae

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            
            - model.beta (int, float): KL divergence weighting term.
            - model.sparse (bool): Whether to enforce sparsity of the encoding distribution.
            - model.threshold (float): Dropout threshold applied to the latent dimensions. Default is 0.
            - encoder.default._target_ (multiviewae.architectures.mlp.VariationalEncoder): Type of encoder class to use.
            - encoder.default.enc_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoding distribution.
            - decoder.default._target_ (multiviewae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
            - decoder.default.init_logvar (int, float): Initial value for log variance of decoder.
            - decoder.default.dec_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoding distribution.

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
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            qz_xs (list): list of encoding dimensions for each view.
        """
        qz_xs = []
        for i in range(self.n_views):
            mu, logvar = self.encoders[i](x[i])
            qz_x = hydra.utils.instantiate(
                eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu, scale=logvar.exp().pow(0.5)+EPS
            )
            qz_xs.append(qz_x)
        return qz_xs

    def decode(self, qz_xs):
        r"""Forward pass through decoder networks. Each latent is passed through all of the decoders.

        Args:
            z (list): list of latent dimensions for each view of type torch.Tensor.

        Returns:
            px_zs (list): A nested list of decoding distributions. The outer list has a n_view element indicating latent dimensions index. 
            The inner list is a n_view element list with the position in the list indicating the decoder index.
        """
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
        r"""Apply encode and decode methods to input data to generate latent dimensions and data reconstructions. 
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding (qz_xs) and decoding (px_zs) distributions.
        """
        qz_xs = self.encode(x)
        px_zs = self.decode(qz_xs)
        fwd_rtn = {"px_zs": px_zs, "qz_xs": qz_xs}
        return fwd_rtn

    def loss_function(self, x, fwd_rtn):
        r"""Calculate mcVAE loss.

        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing each element of the mcVAE loss.
        """
        px_zs = fwd_rtn["px_zs"]
        qz_xs = fwd_rtn["qz_xs"]
        kl = self.calc_kl(qz_xs)
        ll = self.calc_ll(x, px_zs)
        total = kl - ll
        losses = {"loss": total, "kl": kl, "ll": ll}
        return losses

    def calc_kl(self, qz_xs):
        r"""Calculate mcVAE KL-divergence loss.

        Args:
            qz_xs (list): list of encoding distributions.
            
        Returns:
            (torch.Tensor): KL-divergence loss across all views.
        """
        kl = 0
        for qz_x in qz_xs:
            if self.sparse:
                kl += qz_x.sparse_kl_divergence().mean(0).sum()
            else:
                kl += qz_x.kl_divergence(self.prior).mean(0).sum()
        return self.beta * kl

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
            for j in range(self.n_views):
                ll += px_zs[j][i].log_likelihood(x[i]).mean(0).sum() #first index is latent, second index is view
        return ll
