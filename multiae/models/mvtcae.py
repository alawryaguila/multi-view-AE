import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from .layers import Encoder, Decoder
from .utils_deep import Optimisation_VAE
import numpy as np
from ..utils.kl_utils import compute_kl, compute_kl_sparse, compute_ll
from ..utils.calc_utils import ProductOfExperts
from os.path import join
import pytorch_lightning as pl


class MVTCAE(pl.LightningModule, Optimisation_VAE):
    """
    Multi-View Total Correlation Auto-Encoder (MVTCAE) https://proceedings.neurips.cc/paper/2021/hash/65a99bb7a3115fdede20da98b08a370f-Abstract.html
    """

    def __init__(
        self,
        input_dims,
        z_dim=1,
        hidden_layer_dims=[],
        non_linear=False,
        learning_rate=0.001,
        beta=1,
        alpha=0.5,
        trainer_dict=None,
        dist="gaussian",
        **kwargs,
    ):

        """
        :param input_dims: columns of input data e.g. [M1 , M2] where M1 and M2 are number of the columns for views 1 and 2 respectively
        :param z_dim: number of latent vectors
        :param hidden_layer_dims: dimensions of hidden layers for encoder and decoder networks.
        :param non_linear: non-linearity between hidden layers. If True ReLU is applied between hidden layers of encoder and decoder networks
        :param learning_rate: learning rate of optimisers.
        :param beta: weighting factor for Kullback-Leibler divergence term.
        :param dist: Approximate distribution of data for log likelihood calculation. Either 'gaussian', 'MultivariateGaussian' or 'bernoulli'.
        """

        super().__init__()
        self.save_hyperparameters()
        self.model_type = "VAE"
        self.input_dims = input_dims
        hidden_layer_dims = hidden_layer_dims.copy()
        self.z_dim = z_dim
        self.sparse = False
        hidden_layer_dims.append(self.z_dim)
        self.non_linear = non_linear
        self.beta = beta
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.trainer_dict = trainer_dict
        self.dist = dist
        self.variational = True
        self.n_views = len(input_dims)
        self.__dict__.update(kwargs)
        self.encoders = torch.nn.ModuleList(
            [
                Encoder(
                    input_dim=input_dim,
                    hidden_layer_dims=hidden_layer_dims,
                    variational=True,
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
                    variational=True,
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

        losses = {"total": total, "kl_cvib": cvib_kl, "kl_grp": grp_kl, "ll": recon}
        return losses

    def training_step(self, batch, batch_idx, optimizer_idx):
        fwd_return = self.forward(batch)
        loss = self.loss_function(batch, fwd_return)
        self.log(
            f"train_loss", loss["total"], on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"train_kl_cvib_loss",
            loss["kl_cvib"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"train_kl_grpwise_loss",
            loss["kl_grp"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"train_ll_loss", loss["ll"], on_epoch=True, prog_bar=True, logger=True
        )
        return loss["total"]

    def validation_step(self, batch, batch_idx):
        fwd_return = self.forward(batch)
        loss = self.loss_function(batch, fwd_return)
        self.log(f"val_loss", loss["total"], on_epoch=True, prog_bar=True, logger=True)
        self.log(
            f"val_kl_cvib_loss",
            loss["kl_cvib"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"val_kl_grpwise_loss",
            loss["kl_grp"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(f"val_ll_loss", loss["ll"], on_epoch=True, prog_bar=True, logger=True)
        return loss["total"]

    def on_train_end(self):
        self.trainer.save_checkpoint(join(self.output_path, "model.ckpt"))
        torch.save(self, join(self.output_path, "model.pkl"))
