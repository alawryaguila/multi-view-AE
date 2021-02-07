import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .layers import Encoder, Decoder 
from .utils_deep import Optimisation_VAE
import numpy as np

class VAE(nn.Module, Optimisation_VAE):
    
    def __init__(self, input_dims, config):

        ''' 
        Initialise Variational Autoencoder model.

        input_dims: The input data dimension.
        config: Configuration dictionary.

        '''

        super().__init__()
        self._config = config
        self.model_type = 'VAE'
        self.input_dims = input_dims
        self.hidden_layer_dims = config['hidden_layers']
        self.hidden_layer_dims.append(config['latent_size'])
        self.non_linear = config['non_linear']
        self.beta = config['beta']
        self.learning_rate = config['learning_rate']
        self.n_views = len(input_dims)
        self.encoders = torch.nn.ModuleList([Encoder(input_dim = input_dim, hidden_layer_dims=self.hidden_layer_dims, variational=True, non_linear=self.non_linear) for input_dim in self.input_dims])
        self.decoders = torch.nn.ModuleList([Decoder(input_dim = input_dim, hidden_layer_dims=self.hidden_layer_dims, non_linear=self.non_linear) for input_dim in self.input_dims])
        self.optimizers = [torch.optim.Adam(list(self.encoders[i].parameters()) + list(self.decoders[i].parameters()),
                                      lr=self.learning_rate) for i in range(self.n_views)]
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
        for i in range(self.n_views):
            std = torch.exp(0.5*logvar[i])
            eps = torch.randn_like(mu[i])
            z.append(mu[i]+eps*std)
        return z

    def decode(self, z):
        x_same = []
        x_cross = []
        for i in range(self.n_views):
            for j in range(self.n_views):
                mu_out = self.decoders[i](z[j])
                if i == j:
                    x_same.append(mu_out)
                else:
                    x_cross.append(mu_out)
        return x_same, x_cross

    def forward(self, x):
        self.zero_grad()
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_same, x_cross = self.decode(z)
        fwd_rtn = {'x_same': x_same,
                    'x_cross': x_cross,
                    'mu': mu,
                    'logvar': logvar}
        return fwd_rtn

    @staticmethod
    def calc_kl(self, mu, logvar):
        '''
        Implementation from: https://arxiv.org/abs/1312.6114

        '''
        kl = 0
        for i in range(self.n_views):
            kl+= -0.5*torch.sum(1 + logvar[i] - mu[i].pow(2) - logvar[i].exp(), dim=-1).mean(0)
        return self.beta*kl/self.n_views

    @staticmethod
    def recon_loss(self, x, x_same, x_cross):
        recon_loss = 0
        for i in range(self.n_views):
            recon_loss+= torch.mean(((x_same[i] - x[i])**2).sum(dim=-1))
            recon_loss+= torch.mean(((x_cross[i] - x[i])**2).sum(dim=-1))
        return recon_loss/self.n_views/self.n_views

    def loss_function(self, x, fwd_rtn):
        x_same = fwd_rtn['x_same']
        x_cross = fwd_rtn['x_cross']
        mu = fwd_rtn['mu']
        logvar = fwd_rtn['logvar']

        kl = self.calc_kl(self, mu, logvar)
        recon = self.recon_loss(self, x, x_same, x_cross)

        total = kl + recon
        
        losses = {'total': total,
                'kl': kl,
                'reconstruction': recon}
        return losses


__all__ = [
    'VAE'
]