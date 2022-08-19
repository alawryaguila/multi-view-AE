import torch
import hydra

from ..base.constants import MODEL_MEMVAE
from ..base.base_model import BaseModelVAE
from ..base.distributions import Normal
from ..base.exceptions import ModelInputError
from ..base.representations import ProductOfExperts, MeanRepresentation

class me_mVAE(BaseModelVAE):
    """
    Implementation of Multimodal Generative Models for Scalable Weakly-Supervised Learning (https://arxiv.org/abs/1802.05335), MVAE
    Code inspired by: https://github.com/mhw32/multimodal-vae-public

    Loss optimises the ELBO term from the joint posterior distribution, as well as the separate ELBO terms for each view.
    me_mVAE stands for multi ELBO multimodal VAE
    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_MEMVAE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

        self.join_type = self.cfg.model.join_type
        if self.join_type == "PoE":
            self.join_z = ProductOfExperts()
        elif self.join_type == "Mean":
            self.join_z = MeanRepresentation()
        else:
            raise ModelInputError(f"[MVAE] Incorrect join method: {self.join_type}")

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
        mu_out, var_out = self.join_z(mu, var)
        qz_x = hydra.utils.instantiate(
            self.cfg.encoder.enc_dist, loc=mu_out, scale=var_out.pow(0.5)
        )
        return [qz_x]

    def encode_separate(self, x):

        qz_xs = []
        for i in range(self.n_views):
            mu_, logvar_ = self.encoders[i](x[i])
            qz_x = hydra.utils.instantiate(
                self.cfg.encoder.enc_dist, loc=mu_, scale=logvar_.exp().pow(0.5)
            )
            qz_xs.append(qz_x)
        return qz_xs

    def decode(self, qz_x):
        px_zs = []
        for i in range(self.n_views):
            px_z = self.decoders[i](qz_x[0]._sample(training=self._training))
            px_zs.append([px_z])
        return px_zs

    def decode_separate(self, qz_xs):
        px_zs = []
        for i in range(self.n_views):
            px_z = self.decoders[i](qz_xs[i]._sample(training=self._training))
            px_zs.append([px_z])
        return px_zs

    def forward(self, x):
        qz_x = self.encode(x)
        qz_xs = self.encode_separate(x)
        px_zs = self.decode(qz_x)
        px_zss = self.decode_separate(qz_xs)
        fwd_rtn = {"px_zs": px_zs, "px_zss": px_zss, "qz_x": qz_x, "qz_xs": qz_xs}
        return fwd_rtn
        
    def calc_kl(self, qz_xs):
        """
        VAE: Implementation from: https://arxiv.org/abs/1312.6114
        sparse-VAE: Implementation from: https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb
        """
        sh = qz_xs[0].loc.shape
        if isinstance(qz_xs[0], Normal):    # TODO - flexible prior
            prior = torch.distributions.normal.Normal(0,1)
        else:
            prior = torch.distributions.multivariate_normal.MultivariateNormal( \
                        loc=torch.zeros(sh), covariance_matrix=torch.diag_embed(torch.ones(sh)))
        kl = 0
        for i in range(len(qz_xs)):
            if self.sparse:
                kl += qz_xs[i].sparse_kl_divergence().sum(1, keepdims=True).mean(0)
            else:
                kl += qz_xs[i].kl_divergence(prior).sum(1, keepdims=True).mean(0)
        return self.beta * kl

    def calc_ll(self, x, px_zs):
        ll = 0
        for i in range(self.n_views):
            ll += px_zs[i][0].log_likelihood(x[i]).sum(1, keepdims=True).mean(0)
        return ll

    def loss_function(self, x, fwd_rtn):

        px_zs = fwd_rtn["px_zs"]
        qz_x = fwd_rtn["qz_x"]
        px_zss = fwd_rtn["px_zss"]
        qz_xs = fwd_rtn["qz_xs"]


        kl = self.calc_kl(qz_x)
        kl_separate = self.calc_kl(qz_xs)
        ll = self.calc_ll(x, px_zs)
        ll_separate = self.calc_ll(x, px_zss)

        total = kl + kl_separate - ll - ll_separate

        losses = {"loss": total, "kl": kl, "ll": ll, "ll_separate": ll_separate, "kl_separate": kl_separate}
        return losses