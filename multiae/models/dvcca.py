import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .layers import Encoder, Decoder 
from .utils_deep import Optimisation_VAE
import numpy as np
import pytorch_lightning as pl
from os.path import join

class DVCCA(pl.LightningModule, Optimisation_VAE):
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
                private=True,
                **kwargs):
        '''
        Initialise the Deep Variational Canonical Correlation Analysis model

        :param input_dims: columns of input data e.g. [M1 , M2] where M1 and M2 are number of the columns for views 1 and 2 respectively
        :param z_dim: number of latent vectors
        :param hidden_layer_dims: dimensions of hidden layers for encoder and decoder networks.
        :param non_linear: non-linearity between hidden layers. If True ReLU is applied between hidden layers of encoder and decoder networks
        :param learning_rate: learning rate of optimisers.
        :param beta: weighting factor for Kullback-Leibler divergence term.
        :param threshold: Dropout threshold for sparsity constraint on latent representation. If threshold is 0 then there is no sparsity.
        :param private: Label to indicate VCCA or VCCA-private.

        '''

        super().__init__()
        self.save_hyperparameters()
        self.model_type = 'DVCCA'
        self.input_dims = input_dims
        self.hidden_layer_dims = hidden_layer_dims.copy()
        self.z_dim = z_dim
        self.hidden_layer_dims.append(self.z_dim)
        self.non_linear = non_linear
        self.beta = beta
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.trainer_dict = trainer_dict
        if private:
            self.model_type = 'DVCCA_private'
        self.input_dims = input_dims
        self.private = private
        self.n_views = len(input_dims)
        self.encoder = torch.nn.ModuleList([Encoder(input_dim = self.input_dims[0], hidden_layer_dims=self.hidden_layer_dims, variational=True)])
        if private:
            self.private_encoders = torch.nn.ModuleList([Encoder(input_dim = input_dim, hidden_layer_dims=self.hidden_layer_dims, variational=True) for input_dim in self.input_dims])
            self.hidden_layer_dims[-1] = z_dim + z_dim

        self.decoders = torch.nn.ModuleList([Decoder(input_dim = input_dim, hidden_layer_dims=self.hidden_layer_dims, variational=True) for input_dim in self.input_dims])
        if private:
            self.optimizers = [torch.optim.Adam(self.encoder.parameters(),lr=0.001)] + [torch.optim.Adam(list(self.decoders[i].parameters()),
                                      lr=self.learning_rate) for i in range(self.n_views)]
        else:
            self.optimizers = [torch.optim.Adam(self.encoder.parameters(), lr=0.001)] + [torch.optim.Adam(list(self.decoders[i].parameters()),
                                      lr=0.001) for i in range(self.n_views)]
    def encode(self, x):
        mu, logvar = self.encoder[0](x[0])
        if self.private:
            mu_tmp = []
            logvar_tmp = []
            for i in range(self.n_views):
                mu_p, logvar_p = self.private_encoders[i](x[i])
                mu_ = torch.cat((mu, mu_p),1)
                mu_tmp.append(mu_)
                logvar_ = torch.cat((logvar, logvar_p),1)
                logvar_tmp.append(logvar_)
            mu = mu_tmp
            logvar = logvar_tmp
        return mu, logvar
    
    def reparameterise(self, mu, logvar):
        if self.private:
            z = []
            for i in range(self.n_views):
                std = torch.exp(0.5*logvar[i])
                eps = torch.randn_like(mu[i])
                z.append(mu[i]+eps*std)
        else:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(mu)
            z = mu+eps*std
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
        fwd_rtn = {'x_recon': x_recon,
                    'mu': mu,
                    'logvar': logvar}
        return fwd_rtn

    @staticmethod
    def calc_kl(self, mu, logvar):
        '''
        Implementation from: https://arxiv.org/abs/1312.6114

        '''
        kl = 0
        if self.private:
            for i in range(self.n_views):
                kl+= -0.5*torch.sum(1 + logvar[i] - mu[i].pow(2) - logvar[i].exp(), dim=-1)
        else:
            kl+= -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            
        return self.beta*kl

    @staticmethod
    def calc_ll(self, x, x_recon):
        ll = 0
        for i in range(self.n_views):
            ll+= torch.mean(x_recon[i].log_prob(x[i]).sum(dim=1))
        return ll

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn['x_recon']
        mu = fwd_rtn['mu']
        logvar = fwd_rtn['logvar']

        kl = self.calc_kl(self, mu, logvar)
        recon = self.calc_ll(self, x, x_recon)
        total = kl + recon
        losses = {'total': total,
                'kl': kl,
                'll': recon}
        return losses

    def training_step(self, batch, batch_idx, optimizer_idx):
        fwd_return = self.forward(batch)
        loss = self.loss_function(batch, fwd_return)
        self.log(f'train_loss', loss['total'], on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_kl_loss', loss['kl'], on_epoch=True, prog_bar=True, logger=True)
        self.log(f'train_ll_loss', loss['ll'], on_epoch=True, prog_bar=True, logger=True)
        return loss['total']

    def validation_step(self, batch, batch_idx):
        fwd_return = self.forward(batch)
        loss = self.loss_function(batch, fwd_return)
        self.log(f'val_loss', loss['total'], on_epoch=True, prog_bar=True, logger=True)
        self.log(f'val_kl_loss', loss['kl'], on_epoch=True, prog_bar=True, logger=True)
        self.log(f'val_ll_loss', loss['ll'], on_epoch=True, prog_bar=True, logger=True)
        return loss['total']
    
    def on_train_end(self):
        self.trainer.save_checkpoint(join(self.output_path, 'model.ckpt'))