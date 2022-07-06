import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .layers import Encoder, Decoder, Discriminator
from .utils_deep import Optimisation_AAE
import numpy as np
from ..utils.kl_utils import compute_mse
from torch.autograd import Variable
import pytorch_lightning as pl
from os.path import join


class AAE(pl.LightningModule, Optimisation_AAE):
    """
    Multi-view Adversarial Autoencoder model with a separate latent representation for each view.

    """

    def __init__(
        self,
        input_dims,
        z_dim=1,
        hidden_layer_dims=[],
        discriminator_layer_dims=[],
        non_linear=False,
        learning_rate=0.002,
        **kwargs,
    ):

        """
        :param input_dims: columns of input data e.g. [M1 , M2] where M1 and M2 are number of the columns for views 1 and 2 respectively
        :param z_dim: number of latent vectors
        :param hidden_layer_dims: dimensions of hidden layers for encoder and decoder networks.
        :param discriminator_layer_dims: dimensions of hidden layers for encoder and decoder networks.
        :param non_linear: non-linearity between hidden layers. If True ReLU is applied between hidden layers of encoder and decoder networks
        :param learning_rate: learning rate of optimisers.
        """
        super().__init__()
        self.automatic_optimization = False
        self.model_type = "AAE"
        self.input_dims = input_dims
        self.hidden_layer_dims = hidden_layer_dims.copy()
        self.z_dim = z_dim
        self.hidden_layer_dims.append(self.z_dim)
        self.non_linear = non_linear
        self.learning_rate = learning_rate
        self.n_views = len(input_dims)
        self.joint_representation = False
        self.wasserstein = False
        self.variational = False
        self.sparse = False
        self.__dict__.update(kwargs)

        self.encoders = torch.nn.ModuleList(
            [
                Encoder(
                    input_dim=input_dim,
                    hidden_layer_dims=self.hidden_layer_dims,
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
                    hidden_layer_dims=self.hidden_layer_dims,
                    variational=False,
                    non_linear=self.non_linear,
                )
                for input_dim in self.input_dims
            ]
        )
        self.discriminator = Discriminator(
            input_dim=self.z_dim,
            hidden_layer_dims=discriminator_layer_dims,
            output_dim=(self.n_views + 1),
        )
        # TO DO - check discriminator dimensionality is correct (nviews + 1?) and for joint model too

    def configure_optimizers(self):
        optimizers = []
        [
            optimizers.append(
                torch.optim.Adam(
                    list(self.encoders[i].parameters()), lr=self.learning_rate
                )
            )
            for i in range(self.n_views)
        ]
        [
            optimizers.append(
                torch.optim.Adam(
                    list(self.decoders[i].parameters()), lr=self.learning_rate
                )
            )
            for i in range(self.n_views)
        ]
        [
            optimizers.append(
                torch.optim.Adam(
                    list(self.encoders[i].parameters()), lr=self.learning_rate
                )
            )
            for i in range(self.n_views)
        ]
        optimizers.append(
            torch.optim.Adam(
                list(self.discriminator.parameters()), lr=self.learning_rate
            )
        )
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

    def disc(self, z):
        z_real = Variable(torch.randn(z[0].size()[0], self.z_dim) * 1.0).to(self.device)
        d_real = self.discriminator(z_real)
        d_fake = []
        for i in range(self.n_views):
            d = self.discriminator(z[i])
            d_fake.append(d)
        return d_real, d_fake

    def forward_recon(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        fwd_rtn = {"x_recon": x_recon, "z": z}
        return fwd_rtn

    def forward_discrim(self, x):
        [encoder.eval() for encoder in self.encoders]
        z = self.encode(x)
        d_real, d_fake = self.disc(z)
        fwd_rtn = {"d_real": d_real, "d_fake": d_fake, "z": z}
        return fwd_rtn

    def forward_gen(self, x):
        [encoder.train() for encoder in self.encoders]
        self.discriminator.eval()
        z = self.encode(x)
        _, d_fake = self.disc(z)
        fwd_rtn = {"d_fake": d_fake, "z": z}
        return fwd_rtn

    @staticmethod
    def recon_loss(self, x, fwd_rtn):
        x_recon = fwd_rtn["x_recon"]
        recon = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                recon += compute_mse(x[i], x_recon[i][j])
        return recon / self.n_views / self.n_views

    def generator_loss(self, fwd_rtn):
        z = fwd_rtn["z"]
        d_fake = fwd_rtn["d_fake"]
        gen_loss = 0
        label_real = np.zeros((z[0].shape[0], self.n_views + 1))
        label_real[:, 0] = 1
        label_real = torch.FloatTensor(label_real).to(self.device)
        for i in range(self.n_views):
            gen_loss += -torch.mean(label_real * torch.log(d_fake[i] + self.eps))
        return gen_loss / self.n_views

    def discriminator_loss(self, fwd_rtn):
        z = fwd_rtn["z"]
        d_real = fwd_rtn["d_real"]
        d_fake = fwd_rtn["d_fake"]
        disc_loss = 0
        label_real = np.zeros((z[0].shape[0], self.n_views + 1))
        label_real[:, 0] = 1
        label_real = torch.FloatTensor(label_real).to(self.device)

        disc_loss += -torch.mean(label_real * torch.log(d_real + self.eps))
        for i in range(self.n_views):
            label_fake = np.zeros((z[0].shape[0], self.n_views + 1))
            label_fake[:, i + 1] = 1
            label_fake = torch.FloatTensor(label_fake).to(self.device)
            disc_loss += -torch.mean(label_fake * torch.log(d_fake[i] + self.eps))

        return disc_loss / (self.n_views + 1)

    def training_step(self, batch, batch_idx):
        loss = self.optimise_batch(batch)
        self.log(
            f"train_loss", loss["total"], on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"train_recon_loss",
            loss["recon"],
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"train_disc_loss", loss["disc"], on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"train_gen_loss", loss["gen"], on_epoch=True, prog_bar=True, logger=True
        )
        return loss["total"]

    def validation_step(self, batch, batch_idx):
        loss = self.validate_batch(batch)
        self.log(f"val_loss", loss["total"], on_epoch=True, prog_bar=True, logger=True)
        self.log(
            f"val_recon_loss", loss["recon"], on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"val_disc_loss", loss["disc"], on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"val_gen_loss", loss["gen"], on_epoch=True, prog_bar=True, logger=True
        )
        return loss["total"]
