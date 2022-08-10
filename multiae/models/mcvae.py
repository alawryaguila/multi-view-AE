import torch
import hydra

from torch.distributions import Normal

from ..base.constants import MODEL_MCVAE
from ..base.base_model import BaseModelVAE

class mcVAE(BaseModelVAE):
    """
    Multi-view Variational Autoencoder model with a separate latent representation for each view.

    Option to impose sparsity on the latent representations using a Sparse Multi-Channel Variational Autoencoder (http://proceedings.mlr.press/v97/antelmi19a.html)

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
        for i in range(self.n_views):
            px_z = [
                self.decoders[i](qz_x._sample(training=self._training))
                for qz_x in qz_xs
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
        """
        VAE: Implementation from: https://arxiv.org/abs/1312.6114
        sparse-VAE: Implementation from: https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb

        """
        kl = 0
        prior = Normal(0, 1)  # TODO - flexible prior
        for qz_x in qz_xs:
            if self.sparse:
                kl += qz_x.sparse_kl_divergence()#.sum(1, keepdims=True).mean(0)    # TODO: sparse_kl_divergence does not return same shape as kl_divergence
            else:
                kl += qz_x.kl_divergence(prior).sum(1, keepdims=True).mean(0)
        return self.beta * kl

    def calc_ll(self, x, px_zs):
        ll = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                ll += px_zs[i][j].log_likelihood(x[i]).sum(1, keepdims=True).mean(0)
        return ll

    # TODO: this is never used
    # def sample_loc_variance(self, qz_xs):
    #     mu = []
    #     var = []
    #     for qz_x in qz_xs:
    #         mu.append(qz_x.loc)
    #         var.append(qz_x.variance)
    #     return mu, var
