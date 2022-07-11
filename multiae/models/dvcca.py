import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .layers import Encoder, Decoder
from ..base.base_model import BaseModel
import numpy as np
from ..utils.kl_utils import compute_kl, compute_kl_sparse, compute_ll

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
        print(self.hidden_layer_dims)
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
            mu_tmp = []
            logvar_tmp = []
            for i in range(self.n_views):
                mu_p, logvar_p = self.private_encoders[i](x[i])
                mu_ = torch.cat((mu, mu_p), 1)
                mu_tmp.append(mu_)
                logvar_ = torch.cat((logvar, logvar_p), 1)
                logvar_tmp.append(logvar_)
            mu = mu_tmp
            logvar = logvar_tmp
        return mu, logvar

    def reparameterise(self, mu, logvar):
        if self.private:
            z = []
            for i in range(self.n_views):
                std = torch.exp(0.5 * logvar[i])
                eps = torch.randn_like(mu[i])
                z.append(mu[i] + eps * std)
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(mu)
            z = mu + eps * std
        return z

    def decode(self, z):
        x_recon = []
        for i in range(self.n_views):
            if self.private:
                x_out = self.decoders[i](z[i])
            else:
                x_out = self.decoders[i](z)
            x_recon.append(x_out)
        return x_recon

    def forward(self, x):
        self.zero_grad()
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z)
        fwd_rtn = {"x_recon": x_recon, "mu": mu, "logvar": logvar}
        return fwd_rtn

    def calc_kl(self, mu, logvar):
        kl = 0
        if self.private:
            for i in range(self.n_views):
                if self.sparse:
                    kl += compute_kl_sparse(mu[i], logvar[i])
                else:
                    kl += compute_kl(mu[i], logvar[i])
        else:
            if self.sparse:
                compute_kl_sparse(mu, logvar)
            else:
                kl += compute_kl(mu, logvar)
        return self.beta * kl

    def calc_ll(self, x, x_recon):
        ll = 0
        for i in range(self.n_views):
            ll += compute_ll(x[i], x_recon[i], dist=self.dist)
        return ll

    def sample_from_normal(self, normal):
        return normal.loc

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn["x_recon"]
        mu = fwd_rtn["mu"]
        logvar = fwd_rtn["logvar"]

        kl = self.calc_kl(mu, logvar)
        recon = self.calc_ll(x, x_recon)
        total = kl - recon
        losses = {"loss": total, "kl": kl, "ll": recon}
        return losses
