import torch
import hydra

from torch.distributions import Normal

from ..base.constants import MODEL_DVCCA
from ..base.base_model import BaseModelVAE

class DVCCA(BaseModelVAE):
    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        super().__init__(model_name=MODEL_DVCCA,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

    ################################            protected methods
    def _setencoders(self):
        if self.sparse and self.threshold != 0.:
        # if self.threshold != 0: #TODO: threshold takes precedence?
            self.log_alpha = torch.nn.Parameter(
                torch.FloatTensor(1, self.z_dim).normal_(0, 0.01)
            )
        else:
            self.sparse = False
            self.log_alpha = None

        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    self.cfg.encoder,
                    input_dim=self.input_dim[0],
                    z_dim=self.z_dim,
                    sparse=self.sparse,
                    log_alpha=self.log_alpha,
                    _recursive_=False,
                    _convert_="all"
                ) for i in range(2)
            ]
        )
        if self.private:
            self.private_encoders = torch.nn.ModuleList(
                [
                    hydra.utils.instantiate(
                        self.cfg.encoder,
                        input_dim=d,
                        z_dim=self.z_dim,
                        sparse=self.sparse,
                        log_alpha=self.log_alpha,
                        _recursive_=False,
                        _convert_="all"
                    )
                    for d in self.input_dim
                ]
            )
            self.z_dim = self.z_dim + self.z_dim

    def configure_optimizers(self):
        if self.private: # TODO: should use self.private_encoders
            optimizers = [  #TODO: shouldn't this be encoders[0]?
                torch.optim.Adam(self.encoders.parameters(), lr=self.learning_rate)
            ] + [
                torch.optim.Adam(
                    list(self.decoders[i].parameters()), lr=self.learning_rate
                )
                for i in range(self.n_views)
            ]
        else:   # TOD0: this is the same as private=True?
            optimizers = [  #TODO: shouldn't this be encoders[0]?
                torch.optim.Adam(self.encoders.parameters(), lr=self.learning_rate)
            ] + [
                torch.optim.Adam(
                    list(self.decoders[i].parameters()), lr=self.learning_rate
                )
                for i in range(self.n_views)
            ]
        return optimizers

    def encode(self, x):
        mu, logvar = self.encoders[0](x[0])
        if self.private:
            qz_xs = []
            for i in range(self.n_views):
                mu_p, logvar_p = self.private_encoders[i](x[i])
                mu_ = torch.cat((mu, mu_p), 1)
                logvar_ = torch.cat((logvar, logvar_p), 1)
                qz_x = hydra.utils.instantiate(
                    self.cfg.encoder.enc_dist, loc=mu_, scale=logvar_.exp().pow(0.5)
                )
                qz_xs.append(qz_x)
            return qz_xs
        else:
            qz_x = hydra.utils.instantiate(
                self.cfg.encoder.enc_dist, loc=mu, scale=logvar.exp().pow(0.5)
            )
            return [qz_x]

    def decode(self, qz_x):
        px_zs = []
        for i in range(self.n_views):
            if self.private:
                x_out = self.decoders[i](qz_x[i]._sample(training=self._training))
            else:
                x_out = self.decoders[i](qz_x[0]._sample(training=self._training))
            px_zs.append(x_out)
        return px_zs

    def forward(self, x):
        self.zero_grad()
        qz_x = self.encode(x)
        px_zs = self.decode(qz_x)
        fwd_rtn = {"px_zs": px_zs, "qz_x": qz_x}
        return fwd_rtn

    def loss_function(self, x, fwd_rtn):
        px_zs = fwd_rtn["px_zs"]
        qz_x = fwd_rtn["qz_x"]
        kl = self.calc_kl(qz_x)
        ll = self.calc_ll(x, px_zs)
        total = kl - ll
        losses = {"loss": total, "kl": kl, "ll": ll}
        return losses

    def calc_kl(self, qz_x):
        prior = Normal(0, 1)  # TODO - flexible prior
        kl = 0
        if self.private:
            for i in range(self.n_views):
                if self.sparse:
                    kl += qz_x[i].sparse_kl_divergence()#.sum(1, keepdims=True).mean(0) # TODO: sparse_kl_divergence does not return same shape as kl_divergence
                else:
                    kl += qz_x[i].kl_divergence(prior).sum(1, keepdims=True).mean(0)
        else:
            if self.sparse:
                kl += qz_x[0].sparse_kl_divergence()#.sum(1, keepdims=True).mean(0)  # TODO: sparse_kl_divergence does not return same shape as kl_divergence
            else:
                kl += qz_x[0].kl_divergence(prior).sum(1, keepdims=True).mean(0)
        return self.beta * kl

    def calc_ll(self, x, px_zs):
        ll = 0
        for i in range(self.n_views):
            ll += px_zs[i].log_likelihood(x[i]).sum(1, keepdims=True).mean(0)
        return ll
