import torch
import hydra

from torch.distributions import Normal

from ..base.constants import MODEL_MVTCAE
from ..base.base_model import BaseModelVAE
from ..base.representations import ProductOfExperts

class mvtCAE(BaseModelVAE):
    """
    Multi-View Total Correlation Auto-Encoder (MVTCAE) https://proceedings.neurips.cc/paper/2021/hash/65a99bb7a3115fdede20da98b08a370f-Abstract.html
    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        super().__init__(model_name=MODEL_MVTCAE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

    def encode(self, x):
        if self._training:
            qz_xs = []
            for i in range(self.n_views):
                mu, logvar = self.encoders[i](x[i])
                qz_x = hydra.utils.instantiate(
                    self.cfg.encoder.enc_dist, loc=mu, scale=logvar.exp().pow(0.5)
                )
                qz_xs.append(qz_x)
            return qz_xs
        else:
            mu = []
            var = []
            for i in range(self.n_views):
                mu_, logvar_ = self.encoders[i](x[i])
                mu.append(mu_)
                var_ = logvar_.exp()
                var.append(var_)
            mu = torch.stack(mu)
            var = torch.stack(var)
            mu, var = ProductOfExperts()(mu, var)
            qz_x = hydra.utils.instantiate(
                self.cfg.encoder.enc_dist, loc=mu, scale=var.pow(0.5)
            )
            return [qz_x]

    def decode(self, qz_xs):
        if self._training:
            mu = [qz_x.loc for qz_x in qz_xs]
            var = [qz_x.variance for qz_x in qz_xs]
            mu = torch.stack(mu)
            var = torch.stack(var)
            mu, var = ProductOfExperts()(mu, var)
            px_zs = []
            for i in range(self.n_views):
                px_z = self.decoders[i](
                    hydra.utils.instantiate(
                        self.cfg.encoder.enc_dist, loc=mu, scale=var.pow(0.5)
                    ).rsample()
                )
                px_zs.append(px_z)
            return px_zs
        else:
            px_zs = []
            for i in range(self.n_views):
                px_z = self.decoders[i](qz_xs[0].loc)
                px_zs.append(px_z)
            return px_zs

    def forward(self, x):
        qz_xs = self.encode(x)
        px_zs = self.decode(qz_xs)
        fwd_rtn = {"px_zs": px_zs, "qz_xs": qz_xs}
        return fwd_rtn

    def loss_function(self, x, fwd_rtn):
        px_zs = fwd_rtn["px_zs"]
        qz_xs = fwd_rtn["qz_xs"]

        rec_weight = (self.n_views - self.alpha) / self.n_views
        cvib_weight = self.alpha / self.n_views
        vib_weight = 1 - self.alpha

        grp_kl = self.calc_kl_groupwise(qz_xs)
        cvib_kl = self.calc_kl_cvib(qz_xs)
        ll = self.calc_ll(x, px_zs)

        kld_weighted = cvib_weight * cvib_kl + vib_weight * grp_kl
        total = -rec_weight * ll + self.beta * kld_weighted

        losses = {"loss": total, "kl_cvib": cvib_kl, "kl_grp": grp_kl, "ll": ll}
        return losses

    def calc_kl_cvib(self, qz_xs):
        mu = [qz_x.loc for qz_x in qz_xs]
        var = [qz_x.variance for qz_x in qz_xs]
        mu = torch.stack(mu)
        var = torch.stack(var)
        mu, var = ProductOfExperts()(mu, var)
        kl = 0
        for i in range(self.n_views):
            kl += (
                hydra.utils.instantiate(
                    self.cfg.encoder.enc_dist, loc=mu, scale=var.pow(0.5)
                )
                .kl_divergence(qz_xs[i])
                .sum(1, keepdims=True)
                .mean(0)
            )
        return kl

    def calc_kl_groupwise(self, qz_xs):
        mu = [qz_x.loc for qz_x in qz_xs]
        var = [qz_x.variance for qz_x in qz_xs]
        mu = torch.stack(mu)
        var = torch.stack(var)
        mu, var = ProductOfExperts()(mu, var)
        prior = Normal(0, 1)  # TODO - flexible prior
        return (
            hydra.utils.instantiate(
                self.cfg.encoder.enc_dist, loc=mu, scale=var.pow(0.5)
            )
            .kl_divergence(prior)
            .sum(1, keepdims=True)
            .mean(0)
        )

    def calc_ll(self, x, px_zs):
        ll = 0
        for i in range(self.n_views):
            ll += px_zs[i].log_likelihood(x[i]).sum(1, keepdims=True).mean(0)
        return ll
