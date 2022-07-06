import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from os.path import join
from .layers import Encoder, Decoder
from .utils_deep import Optimisation_VAE
import numpy as np
from ..utils.kl_utils import compute_logvar, compute_kl, compute_kl_sparse, compute_ll
from ..utils.calc_utils import ProductOfExperts, MeanRepresentation
import pytorch_lightning as pl


class jointVAE(pl.LightningModule, Optimisation_VAE):
    """
    Multi-view Variational Autoencoder model with a joint latent representation.

    Latent representations are joined either using the Product of Experts (https://arxiv.org/pdf/1410.7827.pdf)
    or the mean of the representations.

    Option to impose sparsity on the latent representations using a Sparse Multi-Channel Variational Autoencoder (http://proceedings.mlr.press/v97/antelmi19a.html)

    """

    def __init__(
        self,
        input_dims,
        z_dim=1,
        hidden_layer_dims=[],
        non_linear=False,
        learning_rate=0.002,
        beta=1,
        threshold=0,
        join_type="Mean",
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
        :param threshold: Dropout threshold for sparsity constraint on latent representation. If threshold is 0 then there is no sparsity.
        :param join_type: How latent representations are combined. Either "Mean" or "PoE".
        :param dist: Approximate distribution of data for log likelihood calculation. Either 'gaussian' or 'bernoulli'.
        """

        super().__init__()
        self.save_hyperparameters()
        self.model_type = "joint_VAE"
        self.input_dims = input_dims
        hidden_layer_dims = hidden_layer_dims.copy()
        self.z_dim = z_dim
        hidden_layer_dims.append(self.z_dim)
        self.non_linear = non_linear
        self.beta = beta
        self.dist = dist
        self.learning_rate = learning_rate
        self.joint_representation = True
        self.variational = True
        self.join_type = join_type
        if self.join_type == "PoE":
            self.join_z = ProductOfExperts()
        elif self.join_type == "Mean":
            self.join_z = MeanRepresentation()
        else:
            print("Incorrect join method")
            exit()
        self.threshold = threshold
        if self.threshold != 0:
            self.sparse = True
            self.model_type = "joint_sparse_VAE"
            self.log_alpha = torch.nn.Parameter(
                torch.FloatTensor(1, self.z_dim).normal_(0, 0.01)
            )
        else:
            self.log_alpha = None
            self.sparse = False
        self.__dict__.update(kwargs)
        self.n_views = len(input_dims)
        self.encoders = torch.nn.ModuleList(
            [
                Encoder(
                    input_dim=input_dim,
                    hidden_layer_dims=hidden_layer_dims,
                    variational=True,
                    non_linear=self.non_linear,
                    sparse=self.sparse,
                    log_alpha=self.log_alpha,
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
        mu = torch.stack(mu)
        logvar = torch.stack(logvar)
        mu, logvar = self.join_z(mu, logvar)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
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
        dropout = self.dropout()
        keep = (dropout < self.threshold).squeeze().cpu()
        if self.joint_representation:
            z[:, ~keep] = 0
        return z

    def calc_kl(self, mu, logvar):
        """
        VAE: Implementation from: https://arxiv.org/abs/1312.6114
        sparse-VAE: Implementation from: https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb
        """
        kl = 0
        if self.sparse:
            kl += compute_kl_sparse(mu, logvar)
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

        kl = self.calc_kl(self, mu, logvar)
        recon = self.calc_ll(self, x, x_recon)

        total = kl - recon
        losses = {"total": total, "kl": kl, "ll": recon}
        return losses

    def training_step(self, batch, batch_idx, optimizer_idx):
        fwd_return = self.forward(batch)
        loss = self.loss_function(batch, fwd_return)
        self.log(
            f"train_loss", loss["total"], on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"train_kl_loss", loss["kl"], on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"train_ll_loss", loss["ll"], on_epoch=True, prog_bar=True, logger=True
        )
        return loss["total"]

    def validation_step(self, batch, batch_idx):
        fwd_return = self.forward(batch)
        loss = self.loss_function(batch, fwd_return)
        self.log(f"val_loss", loss["total"], on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val_kl_loss", loss["kl"], on_epoch=True, prog_bar=True, logger=True)
        self.log(f"val_ll_loss", loss["ll"], on_epoch=True, prog_bar=True, logger=True)
        return loss["total"]

    def on_train_end(self):
        self.trainer.save_checkpoint(join(self.output_path, "model.ckpt"))
        torch.save(self, join(self.output_path, "model.pkl"))
