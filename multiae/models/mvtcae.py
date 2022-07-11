import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from .layers import Encoder, Decoder
from ..base.base_model import BaseModel
import numpy as np
from ..utils.kl_utils import compute_kl, compute_kl_sparse, compute_ll
from ..utils.calc_utils import ProductOfExperts
from os.path import join
import pytorch_lightning as pl

import os
class MVTCAE(BaseModel):
    """
    Multi-View Total Correlation Auto-Encoder (MVTCAE) https://proceedings.neurips.cc/paper/2021/hash/65a99bb7a3115fdede20da98b08a370f-Abstract.html
    """

    def __init__(
        self,
        input_dims,
        expt='MVTCAE',
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
        self.n_views = len(input_dims)

        self.encoders = torch.nn.ModuleList(
            [
                Encoder(
                    input_dim=input_dim,
                    hidden_layer_dims=hidden_layer_dims,
                    variational=self.variational,
                    non_linear=self.non_linear,
                    sparse=False,
                )
                for input_dim in self.input_dims
            ]
        )
        self.decoders = torch.nn.ModuleList(
            [
                Decoder(
                    input_dim=input_dim,
                    hidden_layer_dims=hidden_layer_dims,
                    variational=self.variational,
                    dist=self.dist,
                    non_linear=self.non_linear,
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
        mu = []
        logvar = []
        for i in range(self.n_views):
            mu_, logvar_ = self.encoders[i](x[i])
            mu.append(mu_)
            logvar.append(logvar_)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        mu = torch.stack(mu)
        logvar = torch.stack(logvar)
        mu, logvar = ProductOfExperts()(mu, logvar)
        std = torch.exp(0.5 * logvar)
        # return MultivariateNormal(mu, torch.diag_embed(std)).rsample()
        eps = torch.randn_like(mu)
        return mu + eps * std

    def decode(self, z):
        x_recon = []
        for i in range(self.n_views):
            mu_out = self.decoders[i](z)
            x_recon.append(mu_out)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z)
        fwd_rtn = {"x_recon": x_recon, "mu": mu, "logvar": logvar}
        return fwd_rtn

    def calc_kl_cvib(self, mu, logvar):
        mugrp = torch.stack(mu)
        logvargrp = torch.stack(logvar)
        mugrp, logvargrp = ProductOfExperts()(mugrp, logvargrp)
        kl = 0
        for i in range(self.n_views):
            kl += compute_kl(mugrp, logvargrp, mu[i], logvar[i])
        return kl

    def calc_kl_groupwise(self, mu, logvar):
        mu = torch.stack(mu)
        logvar = torch.stack(logvar)
        mu, logvar = ProductOfExperts()(mu, logvar)
        return compute_kl(mu, logvar)

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

        rec_weight = (self.n_views - self.alpha) / self.n_views
        cvib_weight = self.alpha / self.n_views
        vib_weight = 1 - self.alpha

        grp_kl = self.calc_kl_groupwise(mu, logvar)
        cvib_kl = self.calc_kl_cvib(mu, logvar)
        recon = self.calc_ll(x, x_recon)

        kld_weighted = cvib_weight * cvib_kl + vib_weight * grp_kl
        total = -rec_weight * recon + self.beta * kld_weighted

        losses = {"loss": total, "kl_cvib": cvib_kl, "kl_grp": grp_kl, "ll": recon}
        return losses
