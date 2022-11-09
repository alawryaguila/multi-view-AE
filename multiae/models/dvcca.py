import torch
import hydra
from ..base.constants import MODEL_DVCCA
from ..base.base_model import BaseModelVAE

class DVCCA(BaseModelVAE):
    r"""Deep Variational Canonical Correlation Analysis (DVCCA).
    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            model.beta (int, float): KL divergence weighting term.
            model.private (bool): Whether to include private view-specific latent dimensions.
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
    Wang, Weiran & Lee, Honglak & Livescu, Karen. (2016). Deep Variational Canonical Correlation Analysis.


    """
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
            self.log_alpha = torch.nn.Parameter(
                torch.FloatTensor(1, self.z_dim).normal_(0, 0.01)
            )
        else:
            self.sparse = False
            self.log_alpha = None

        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    self.cfg.encoder.default,   #TODO: should use enc0?
                    input_dim=self.input_dim[0],
                    z_dim=self.z_dim,
                    sparse=self.sparse,
                    log_alpha=self.log_alpha,
                    _recursive_=False,
                    _convert_="all"
                )
            ]
        )

        if self.private:

            self.private_encoders = torch.nn.ModuleList(
                [
                    hydra.utils.instantiate(
                        eval(f"self.cfg.encoder.enc{i}"),
                        input_dim=d,
                        z_dim=self.z_dim,
                        sparse=self.sparse,
                        log_alpha=self.log_alpha,
                        _recursive_=False,
                        _convert_="all"
                    )
                    for i, d in enumerate(self.input_dim)
                ]
            )
            self.z_dim = self.z_dim + self.z_dim
            if self.sparse and self.threshold != 0.:

                self.log_alpha = torch.nn.Parameter(
                    torch.FloatTensor(1, self.z_dim).normal_(0, 0.01)
                )

    def configure_optimizers(self):
        if self.private:
            optimizers = [
                torch.optim.Adam(self.encoders[0].parameters(), lr=self.learning_rate)
            ] + [
                torch.optim.Adam(
                    list(self.private_encoders[i].parameters()), lr=self.learning_rate
                )
                for i in range(self.n_views)
            ] + [
                torch.optim.Adam(
                    list(self.decoders[i].parameters()), lr=self.learning_rate
                )
                for i in range(self.n_views)
            ]
        else:
            optimizers = [
                torch.optim.Adam(self.encoders[0].parameters(), lr=self.learning_rate)
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
                    eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_, scale=logvar_.exp().pow(0.5)
                )
                qz_xs.append(qz_x)
            return qz_xs
        else:
            qz_x = hydra.utils.instantiate( #TODO: should use enc0?
                self.cfg.encoder.default.enc_dist, loc=mu, scale=logvar.exp().pow(0.5)
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
        return [px_zs]

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
        kl = 0
        if self.private:
            n = self.n_views
        else:
            n = 1
        for i in range(n):
            if self.sparse:
                kl += qz_x[i].sparse_kl_divergence().mean(0).sum()
            else:
                kl += qz_x[i].kl_divergence(self.prior).mean(0).sum()
        return self.beta * kl

    def calc_ll(self, x, px_zs):
        ll = 0
        for i in range(self.n_views):
            ll += px_zs[0][i].log_likelihood(x[i]).mean(0).sum()  #first index is latent, second index is view
        return ll
