import torch
from .layers import Encoder, Decoder
from ..base.base_model import BaseModel
from torch.distributions import Normal
from ..utils.calc_utils import update_dict
import hydra 


class mcVAE(BaseModel):
    """
    Multi-view Variational Autoencoder model with a separate latent representation for each view.

    Option to impose sparsity on the latent representations using a Sparse Multi-Channel Variational Autoencoder (http://proceedings.mlr.press/v97/antelmi19a.html)

    """

    def __init__(
        self,
        input_dims,
        expt='mcVAE',
        **kwargs,
    ):

        super().__init__(expt=expt)

        self.save_hyperparameters()

        self.__dict__.update(self.cfg.model)
        self.__dict__.update(kwargs)

        self.cfg.encoder = update_dict(self.cfg.encoder, kwargs)
        self.cfg.decoder = update_dict(self.cfg.decoder, kwargs)

        self.model_type = expt
        self.input_dims = input_dims
        self.n_views = len(input_dims)

        if self.threshold != 0:
            self.sparse = True
            self.model_type = "sparse_VAE"
            self.log_alpha = torch.nn.Parameter(
                torch.FloatTensor(1, self.z_dim).normal_(0, 0.01)
            )
        else:
            self.log_alpha = None
            self.sparse = False
        self.n_views = len(input_dims)
        self.__dict__.update(kwargs)

        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(self.cfg.encoder,
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
                hydra.utils.instantiate(self.cfg.decoder,
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
            qz_x = hydra.utils.instantiate(self.cfg.encoder.enc_dist, loc=mu, scale=logvar.exp().pow(0.5))
            qz_xs.append(qz_x)
        return qz_xs

    def sample_loc_variance(self, qz_xs):
        mu = []
        var = []
        for qz_x in qz_xs:
            mu.append(qz_x.loc)
            var.append(qz_x.variance)
        return mu, var
        
    def decode(self, qz_xs):
        px_zs = []
        for i in range(self.n_views):
            px_z = [self.decoders[i](qz_x._sample(training=self._training)) for qz_x in qz_xs]
            px_zs.append(px_z)
            del px_z
        return px_zs

    def forward(self, x):
        qz_xs = self.encode(x)
        px_zs = self.decode(qz_xs)
        fwd_rtn = {"px_zs": px_zs, "qz_xs": qz_xs}
        return fwd_rtn

    def dropout(self):
        """
        Implementation from: https://github.com/ggbioing/mcvae
        """
        if self.sparse:
            alpha = torch.exp(self.log_alpha.detach())
            return alpha / (alpha + 1)
        else:
            raise NotImplementedError

    def apply_threshold(self, z):
        """
        Implementation from: https://github.com/ggbioing/mcvae
        """
    
        assert self.threshold <= 1.0
        keep = (self.dropout() < self.threshold).squeeze().cpu()
        z_keep = []
        for _ in z:
            _ = _._sample()
            _[:, ~keep] = 0
            z_keep.append(_)
            del _
        return hydra.utils.instantiate(self.cfg.encoder.enc_dist, loc=z_keep)

    def calc_kl(self, qz_xs):
        """
        VAE: Implementation from: https://arxiv.org/abs/1312.6114
        sparse-VAE: Implementation from: https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb

        """
        kl = 0
        prior = Normal(0, 1) #TODO - flexible prior
        for qz_x in qz_xs:
            if self.sparse:
                kl += qz_x.sparse_kl_divergence().sum(1, keepdims=True).mean(0) 
            else:
                kl += qz_x.kl_divergence(prior).sum(1, keepdims=True).mean(0)
        return self.beta * kl

    def calc_ll(self, x, px_zs):
        ll = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                ll += px_zs[i][j].log_likelihood(x[i]).sum(1, keepdims=True).mean(0)
        return ll

    def sample_from_dist(self, dist):
        return dist._sample()

    def loss_function(self, x, fwd_rtn):
        px_zs = fwd_rtn["px_zs"]
        qz_xs = fwd_rtn["qz_xs"]
        kl = self.calc_kl(qz_xs)
        ll = self.calc_ll(x, px_zs)
        total = kl - ll
        losses = {"loss": total, "kl": kl, "ll": ll}
        return losses
