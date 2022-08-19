import math
import torch
import hydra

from ..base.constants import MODEL_MMVAE
from ..base.base_model import BaseModelVAE
from ..base.distributions import Normal, MultivariateNormal

class mmVAE(BaseModelVAE):
    """
    Multi-view Variational Autoencoder model using Mixture of Experts approach (https://arxiv.org/abs/1911.03393).
    Code is based on: https://github.com/iffsid/mmvae

    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_MMVAE,
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
                self.decoders[j](qz_xs[i].rsample(torch.Size([self.K])))
                for j in range(self.n_views)
            ]
            px_zs.append(
                px_z
            )  # TODO: this is other way around to other multiautoencoder models - FIX
            del px_z
        return px_zs

    def forward(self, x):
        qz_xs = self.encode(x)
        px_zs = self.decode(qz_xs)
        return {"qz_xs": qz_xs, "px_zs": px_zs}

    def loss_function(self, x, fwd_rtn):
        qz_xs, px_zs = fwd_rtn["qz_xs"], fwd_rtn["px_zs"]
        total = -self.moe_iwae(x, qz_xs, px_zs)
        losses = {"loss": total}
        return losses

    def moe_iwae(self, x, qz_xs, px_zs):
        lws = []
        zss = []
        for i in range(self.n_views):
            zs = qz_xs[i].rsample(torch.Size([self.K]))
            zss.append(zs)

        # TODO: please fix this. MultivariateNormal outputs different shape from Normal for log_prob() and kl_divergence()
        for r, qz_x in enumerate(qz_xs):
            if isinstance(qz_xs[0], Normal):
                lpz = Normal(loc=0,scale=1).log_likelihood(zss[r]).sum(-1)
                lqz_x = self.log_mean_exp(
                    torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs])
                )  # summing over M modalities for each z to create q(z|x1:M)
            else:   # TODO: hack
                sh = zss[r].shape
                lpz = MultivariateNormal(loc=torch.zeros(sh), scale=torch.ones(sh)).log_likelihood(zss[r]).reshape((sh[0], sh[1], 1)).sum(-1)
                lqz_x = self.log_mean_exp(
                    torch.stack([qz_x.log_prob(zss[r]).reshape((sh[0], sh[1], 1)).sum(-1) for qz_x in qz_xs])
                )  # summing over M modalities for each z to create q(z|x1:M)

            lpx_z = [
                px_z.log_likelihood(x[d]).view(*px_z._sample().size()[:2], -1).sum(-1)
                for d, px_z in enumerate(px_zs[r])
            ]  # summing over each decoder
            lpx_z = torch.stack(lpx_z).sum(0)
            lw = lpz + lpx_z - lqz_x
            lws.append(lw)
        return (
            self.log_mean_exp(torch.stack(lws), dim=1).mean(0).sum()
        )  # looser iwae bound where have #TODO: what does this comment mean?

    def log_mean_exp(self, value, dim=0, keepdim=False):
        return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))
