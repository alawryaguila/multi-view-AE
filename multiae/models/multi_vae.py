import torch
import torch.nn as nn
import torch.nn.functional as F

# import pytorch_lightning as pl
from torch.distributions import Normal
from .layers import Encoder, Decoder
from .utils_deep import Optimisation_VAE
import numpy as np
from ..utils.kl_utils import compute_kl, compute_kl_sparse, compute_ll
from os.path import join
import pytorch_lightning as pl


class multiVAE(pl.LightningModule, Optimisation_VAE):
    """
    Multi-view Variational Autoencoder model with a separate latent representation for each view.

    Option to impose sparsity on the latent representations using a Sparse Multi-Channel Variational Autoencoder (http://proceedings.mlr.press/v97/antelmi19a.html)

    """

    def __init__(
        self,
        input_dims,
        z_dim=1,
        hidden_layer_dims=[],
        non_linear=False,
        learning_rate=0.001,
        beta=1,
        threshold=0,
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
        :param threshold: Dropout threshold for sparsity constraint on latent representation. If threshold is 0 then there is no sparsity.
        :param dist: Approximate distribution of data for log likelihood calculation. Either 'gaussian' or 'bernoulli'.
        """

        super().__init__()
        self.save_hyperparameters()
        self.model_type = "VAE"
        self.input_dims = input_dims
        hidden_layer_dims = hidden_layer_dims.copy()
        self.z_dim = z_dim
        hidden_layer_dims.append(self.z_dim)
        self.non_linear = non_linear
        self.beta = beta
        self.learning_rate = learning_rate
        self.joint_representation = False
        self.threshold = threshold
        self.trainer_dict = trainer_dict
        self.dist = dist
        self.variational = True
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
        return mu, logvar

    def reparameterise(self, mu, logvar):
        z = []
        for i in range(len(mu)):
            std = torch.exp(0.5 * logvar[i])
            eps = torch.randn_like(mu[i])
            z.append(mu[i] + eps * std)
        return z

    def decode(self, z):
        x_recon = []
        for i in range(self.n_views):
            temp_recon = [self.decoders[i](z[j]) for j in range(self.n_views)]
            x_recon.append(temp_recon)
            del temp_recon
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
        keep = (self.dropout() < self.threshold).squeeze().cpu()
        z_keep = []
        for _ in z:
            _[:, ~keep] = 0
            z_keep.append(_)
            del _
        return z_keep

    def calc_kl(self, mu, logvar):
        """
        VAE: Implementation from: https://arxiv.org/abs/1312.6114
        sparse-VAE: Implementation from: https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb

        """
        kl = 0
        for i in range(self.n_views):
            if self.sparse:
                kl += compute_kl_sparse(mu[i], logvar[i])
            else:
                kl += compute_kl(mu[i], logvar[i])
        return self.beta * kl

    def calc_ll(self, x, x_recon):
        ll = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                ll += compute_ll(x[i], x_recon[i][j], dist=self.dist)
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
