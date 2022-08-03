import torch
from torch.distributions import Normal
from ..base.base_model import BaseModel
from ..utils.calc_utils import update_dict
import math
import hydra


class mmVAE(BaseModel):
    """
    Multi-view Variational Autoencoder model using Mixture of Experts approach (https://arxiv.org/abs/1911.03393).
    Code is based on: https://github.com/iffsid/mmvae

    """

    def __init__(
        self,
        input_dims,
        model="MMVAE",
        network=None,
        **kwargs,
    ):

        super().__init__(model=model, network=network)

        self.save_hyperparameters()

        self.__dict__.update(self.cfg.model)
        self.__dict__.update(kwargs)

        self.cfg.encoder = update_dict(self.cfg.encoder, kwargs)
        self.cfg.decoder = update_dict(self.cfg.decoder, kwargs)

        self.model_type = model
        self.input_dims = input_dims
        self.n_views = len(input_dims)

        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    self.cfg.encoder,
                    _recursive_=False,
                    input_dim=input_dim,
                    z_dim=self.z_dim,
                    sparse=self.sparse,
                    log_alpha=self.log_alpha,
                )
                for input_dim in self.input_dims
            ]
        )
        self.decoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    self.cfg.decoder,
                    _recursive_=False,
                    input_dim=input_dim,
                    z_dim=self.z_dim,
                )
                for input_dim in self.input_dims
            ]
        )

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                list(self.encoders[i].parameters())
                + list(self.decoders[i].parameters()),
                lr=self.learning_rate,
            )
            for i in range(self.n_views)
        ]
        return optimizers

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
                self.decoders[j](qz_xs[i].rsample(torch.Size([self.K])))
                for j in range(self.n_views)
            ]
            px_zs.append(
                px_z
            )  # NOTE: this is other way around to other multiautoencoder models - FIX
            del px_z
        return px_zs

    def forward(self, x):
        qz_xs = self.encode(x)
        px_zs = self.decode(qz_xs)
        return {"qz_xs": qz_xs, "px_zs": px_zs}

    def sample_loc_variance(self, qz_xs):
        mu = []
        var = []
        for qz_x in qz_xs:
            mu.append(qz_x.loc)
            var.append(qz_x.variance)
        return mu, var

    def moe_iwae(self, x, qz_xs, px_zs):
        lws = []
        zss = []
        for i in range(self.n_views):
            zs = qz_xs[i].rsample(torch.Size([self.K]))
            zss.append(zs)
        for r, qz_x in enumerate(qz_xs):
            lpz = Normal(loc=0, scale=1).log_prob(zss[r]).sum(-1)  # TODO flexible prior
            lqz_x = self.log_mean_exp(
                torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs])
            )  # summing over M modalities for each z to create q(z|x1:M)
            lpx_z = [
                px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1).sum(-1)
                for d, px_z in enumerate(px_zs[r])
            ]  # summing over each decoder
            lpx_z = torch.stack(lpx_z).sum(0)
            lw = lpz + lpx_z - lqz_x
            lws.append(lw)
        return (
            self.log_mean_exp(torch.stack(lws), dim=1).mean(0).sum()
        )  # looser iwae bound where have

    def sample_from_dist(self, dist):
        return dist._sample()

    def log_mean_exp(self, value, dim=0, keepdim=False):
        return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))

    def loss_function(self, x, fwd_rtn):
        qz_xs, px_zs = fwd_rtn["qz_xs"], fwd_rtn["px_zs"]
        total = -self.moe_iwae(x, qz_xs, px_zs)
        losses = {"loss": total}
        return losses
