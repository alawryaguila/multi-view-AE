import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .layers import Encoder, Decoder
from .utils_deep import Optimisation_VAE
import numpy as np
from ..utils.kl_utils import compute_mse
import pytorch_lightning as pl
from os.path import join


class AE(pl.LightningModule, Optimisation_VAE):
    def __init__(
        self,
        input_dims,
        z_dim=1,
        hidden_layer_dims=[],
        non_linear=False,
        learning_rate=0.001,
        SNP_model=False,
        trainer_dict=None,
        **kwargs,
    ):

        """
        :param input_dims: columns of input data e.g. [M1 , M2] where M1 and M2 are number of the columns for views 1 and 2 respectively
        :param z_dim: number of latent vectors
        :param hidden_layer_dims: dimensions of hidden layers for encoder and decoder networks.
        :param non_linear: non-linearity between hidden layers. If True ReLU is applied between hidden layers of encoder and decoder networks
        :param learning_rate: learning rate of optimisers.
        :param SNP_model: Whether model will be used for SNP data - parameter will be removed soon.
        """

        super().__init__()
        self.model_type = "AE"
        self.input_dims = input_dims
        hidden_layer_dims = hidden_layer_dims.copy()
        self.z_dim = z_dim
        hidden_layer_dims.append(self.z_dim)
        self.non_linear = non_linear
        self.learning_rate = learning_rate
        self.SNP_model = SNP_model
        self.joint_representation = False
        self.trainer_dict = trainer_dict
        self.variational = False
        self.sparse = False
        self.n_views = len(input_dims)
        self.__dict__.update(kwargs)
        self.encoders = torch.nn.ModuleList(
            [
                Encoder(
                    input_dim=input_dim,
                    hidden_layer_dims=hidden_layer_dims,
                    variational=False,
                    non_linear=self.non_linear,
                )
                for input_dim in self.input_dims
            ]
        )
        self.decoders = torch.nn.ModuleList(
            [
                Decoder(
                    input_dim=input_dim,
                    hidden_layer_dims=hidden_layer_dims,
                    variational=False,
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
        z = []
        for i in range(self.n_views):
            z_ = self.encoders[i](x[i])
            z.append(z_)
        return z

    def decode(self, z):
        x_recon = []
        for i in range(self.n_views):
            temp_recon = [self.decoders[i](z[j]) for j in range(self.n_views)]
            x_recon.append(temp_recon)
            del temp_recon
        return x_recon

    def forward(self, x):
        self.zero_grad()
        z = self.encode(x)
        x_recon = self.decode(z)
        fwd_rtn = {"x_recon": x_recon, "z": z}
        return fwd_rtn

    def recon_loss(self, x, x_recon):
        recon = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                recon += compute_mse(x[i], x_recon[i][j])
        return recon / self.n_views / self.n_views

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn["x_recon"]
        z = fwd_rtn["z"]
        recon = self.recon_loss(self, x, x_recon)
        losses = {"total": recon}
        return losses

    def training_step(self, batch, batch_idx, optimizer_idx):

        fwd_return = self.forward(batch)
        loss = self.loss_function(batch, fwd_return)
        self.log(
            f"train_loss", loss["total"], on_epoch=True, prog_bar=True, logger=True
        )
        return loss["total"]

    def validation_step(self, batch, batch_idx):
        fwd_return = self.forward(batch)
        loss = self.loss_function(batch, fwd_return)
        self.log(f"val_loss", loss["total"], on_epoch=True, prog_bar=True, logger=True)
        return loss["total"]

    def on_train_end(self):
        self.trainer.save_checkpoint(join(self.output_path, "model.ckpt"))
