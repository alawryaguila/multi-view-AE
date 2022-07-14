import torch
from torch.distributions import Normal
from .layers import Encoder, Decoder
from ..base.base_model import BaseModel
import hydra 

class DVCCA(BaseModel):
    def __init__(
        self,
        input_dims,
        expt='DVCCA',
        **kwargs,
    ):

        super().__init__(expt=expt)

        self.save_hyperparameters()

        self.__dict__.update(self.cfg.model)
        self.__dict__.update(kwargs)

        self.model_type = expt
        self.input_dims = input_dims
        hidden_layer_dims = self.hidden_layer_dims.copy()  
        hidden_layer_dims.append(self.z_dim)
        self.hidden_layer_dims = hidden_layer_dims
        self.n_views = len(input_dims)

        if self.threshold != 0:
            self.sparse = True
            self.log_alpha = torch.nn.Parameter(
                torch.FloatTensor(1, self.z_dim).normal_(0, 0.01)
            )
        else:
            self.log_alpha = None
            self.sparse = False

        self.encoder = torch.nn.ModuleList(
            [
                Encoder(
                    input_dim=self.input_dims[0],
                    hidden_layer_dims=self.hidden_layer_dims,
                    sparse=self.sparse,
                    variational=True,
                )
            ]
        )
        if self.private:
            self.private_encoders = torch.nn.ModuleList(
                [
                    Encoder(
                        input_dim=input_dim,
                        hidden_layer_dims=self.hidden_layer_dims,
                        sparse=self.sparse,
                        variational=True,
                    )
                    for input_dim in self.input_dims
                ]
            )
            self.hidden_layer_dims[-1] = self.z_dim + self.z_dim

        self.decoders = torch.nn.ModuleList(
            [
                Decoder(
                    input_dim=input_dim,
                    hidden_layer_dims=self.hidden_layer_dims,
                    dist=self.dist,
                    variational=True,
                )
                for input_dim in self.input_dims
            ]
        )

    def configure_optimizers(self):
        if self.private:
            optimizers = [torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)] + [
                torch.optim.Adam(
                    list(self.decoders[i].parameters()), lr=self.learning_rate
                )
                for i in range(self.n_views)
            ]
        else:
            optimizers = [torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)] + [
                torch.optim.Adam(list(self.decoders[i].parameters()), lr=self.learning_rate)
                for i in range(self.n_views)
            ]
        return optimizers

    def encode(self, x):
        mu, logvar = self.encoder[0](x[0])
        if self.private:
            qz_xs = []
            for i in range(self.n_views):
                mu_p, logvar_p = self.private_encoders[i](x[i])
                mu_ = torch.cat((mu, mu_p), 1)
                logvar_ = torch.cat((logvar, logvar_p), 1)
                qz_x = hydra.utils.instantiate(self.enc_dist, loc=mu_, scale=logvar_.exp().pow(0.5))
                qz_xs.append(qz_x)
            return qz_xs
        else:
            qz_x = hydra.utils.instantiate(self.enc_dist, loc=mu, scale=logvar.exp().pow(0.5))
            return qz_x

    def decode(self, qz_x):
        px_zs = []
        for i in range(self.n_views):
            if self.private:
                x_out = self.decoders[i](qz_x[i].rsample())
            else:
                x_out = self.decoders[i](qz_x.rsample())
            px_zs.append(x_out)
        return px_zs

    def forward(self, x):
        self.zero_grad()
        qz_x = self.encode(x)
        px_zs = self.decode(qz_x)
        fwd_rtn = {"px_zs": px_zs, "qz_x": qz_x}
        return fwd_rtn

    def calc_kl(self, qz_x):
        prior = Normal(0, 1) #TODO - flexible prior
        kl = 0
        if self.private:
            for i in range(self.n_views):
                if self.sparse:
                    kl += qz_x[i].sparse_kl_divergence().sum(1, keepdims=True).mean(0) 
                else:
                    kl += qz_x[i].kl_divergence(prior).sum(1, keepdims=True).mean(0)
        else:
            if self.sparse:
                kl += qz_x.sparse_kl_divergence().sum(1, keepdims=True).mean(0) 
            else:
                kl += qz_x.kl_divergence(prior).sum(1, keepdims=True).mean(0)
        return self.beta * kl

    def calc_ll(self, x, px_zs):
        ll = 0
        for i in range(self.n_views):
            ll += px_zs[i].log_likelihood(x[i]).sum(1, keepdims=True).mean(0)
        return ll

    def sample_from_dist(self, dist):
        return dist._sample()

    def loss_function(self, x, fwd_rtn):
        px_zs = fwd_rtn["px_zs"]
        qz_x = fwd_rtn["qz_x"]
        kl = self.calc_kl(qz_x)
        ll = self.calc_ll(x, px_zs)
        total = kl - ll
        losses = {"loss": total, "kl": kl, "ll": ll}
        return losses
