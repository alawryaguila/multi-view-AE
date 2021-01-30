import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .layers import Encoder, Decoder 
from .utils_deep import Optimisation_AE
import numpy as np

#TO DO - device comes from config file

DEVICE = torch.device("cuda")

class AE(nn.Module, Optimisation_AE):
    
    def __init__(self, input_dims, config):

        '''
        Initialise Autoencoder model.

        input_dims: The input data dimension.
        config: Configuration dictionary.
        beta: KL weight.
        
        '''

        super().__init__()
        self._config = config
        self.model_type = 'AE'
        self.input_dims = input_dims
        self.hidden_layer_dims = config['hidden_layers']
        self.hidden_layer_dims.append(config['latent_size'])
        self.non_linear = config['non_linear']
        self.beta = config['beta']
        self.n_views = len(input_dims)
        self.encoders = torch.nn.ModuleList([Encoder(input_dim = input_dim, hidden_layer_dims=self.hidden_layer_dims, variational=False, non_linear=self.non_linear) for input_dim in self.input_dims])
        self.decoders = torch.nn.ModuleList([Decoder(input_dim = input_dim, hidden_layer_dims=self.hidden_layer_dims, non_linear=self.non_linear) for input_dim in self.input_dims])
        self.optimizers = [torch.optim.Adam(list(self.encoders[i].parameters()) + list(self.decoders[i].parameters()),
                                      lr=0.001) for i in range(self.n_views)]
    def encode(self, x):
        z = []
        for i in range(self.n_views):
            z_ = self.encoders[i](x[i])
            z.append(z_)

        return z
    

    def decode(self, z):
        x_same = []
        x_cross = []
        for i in range(self.n_views):
            for j in range(self.n_views):
                z_out = self.decoders[i](z[j])
                if i == j:
                    x_same.append(z_out)
                else:
                    x_cross.append(z_out)
        return x_same, x_cross

    def forward(self, x):
        self.zero_grad()
        z = self.encode(x)
        x_same, x_cross = self.decode(z)
        fwd_rtn = {'x_same': x_same,
                    'x_cross': x_cross,
                    'z': z}
        return fwd_rtn

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
        z = fwd_rtn['z']
        recon = self.recon_loss(self, x, x_same, x_cross)

        losses = {'total': recon}
        return losses


__all__ = [
    'AE'
]