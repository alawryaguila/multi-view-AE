import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .layers import Encoder, Decoder 
from .utils_deep import Optimisation_VAE
import numpy as np
from ..utils.kl_utils import compute_logvar, compute_kl, compute_kl_sparse

class VAE(nn.Module, Optimisation_VAE):
    
    def __init__(
                self, 
                input_dims, 
                config,
                initial_weights=None):

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
        self.z_dim = config['latent_size']
        self.hidden_layer_dims.append(self.z_dim)
        self.non_linear = config['non_linear']
        self.beta = config['beta']
        self.learning_rate = config['learning_rate']
        self.sparse = config['sparse']
        if self.sparse == True:
            self.model_type = 'sparse_VAE'

        self.initial_weights = initial_weights
        self.joint_representation = False
        if self.sparse:
            self.threshold = config['dropout_threshold']
            self.log_alpha = torch.nn.Parameter(torch.FloatTensor(1, self.z_dim).normal_(0,0.01))
        else:
            self.log_alpha = None
        self.n_views = len(input_dims)
        self.encoders = torch.nn.ModuleList([Encoder(input_dim=input_dim, hidden_layer_dims=self.hidden_layer_dims, variational=True, non_linear=self.non_linear, sparse=self.sparse, log_alpha=self.log_alpha) for input_dim in self.input_dims])
        self.decoders = torch.nn.ModuleList([Decoder(input_dim=input_dim, hidden_layer_dims=self.hidden_layer_dims, variational=True, non_linear=self.non_linear) for input_dim in self.input_dims])
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
        for i in range(len(mu)):
            std = torch.exp(0.5*logvar[i])
            eps = torch.randn_like(mu[i])
            z.append(mu[i]+eps*std)
        return z

    def decode_old(self,z):
        x_same = []
        x_cross = []
        for i in range(self.n_views):
            for j in range(self.n_views):
                mu_out = self.decoders[j](z[i]) #TO CHECK: why does the order of i and j matter here?
                #maybe its a problem relating to not clearing the weights of the network or something
                #ie problem using the same decoder twice in a row
                #think the problem was actually that I didn't have the i and j lined up for the calc_ll
                if i == j:
                    x_same.append(mu_out)
                else:
                    x_cross.append(mu_out) #now this x_cross is in the wrong order
                del mu_out
        return [x_same, x_cross]

    def decode(self, z):
        x_recon = []
        for i in range(self.n_views):
            temp_recon = [self.decoders[i](z[j]) for j in range(self.n_views)]
            x_recon.append(temp_recon)
            del temp_recon 
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

    def dropout(self):
        '''
        Implementation from: https://github.com/ggbioing/mcvae
        '''
        if self.sparse:
            alpha = torch.exp(self.log_alpha.detach())
            return alpha / (alpha + 1)
        else:
            raise NotImplementedError

    def apply_threshold(self, z):
        '''
        Implementation from: https://github.com/ggbioing/mcvae
        '''
        assert self.threshold <= 1.0
        #dropout = self.dropout()
        keep = (self.dropout() < self.threshold).squeeze().cpu()
        z_keep = []
        if self.joint_representation:
            z[:,~keep] = 0
        else:
            for _ in z:
                _[:, ~keep] = 0
                z_keep.append(_)
                del _
        return z

    @staticmethod
    def calc_kl(self, mu, logvar):
        '''
        VAE: Implementation from: https://arxiv.org/abs/1312.6114
        sparse-VAE: Implementation from: https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb

        '''
        kl = 0
        for i in range(self.n_views):
            if self.sparse:
                kl+= compute_kl_sparse(mu[i], logvar[i])
            else:
                kl+= compute_kl(mu[i], logvar[i])
        return self.beta*kl

    @staticmethod
    def calc_ll(self, x, x_recon):
        ll = 0    
        for i in range(self.n_views):
            for j in range(self.n_views):
                    ll+= x_recon[i][j].log_prob(x[i]).sum(1, keepdims=True).mean(0) 
                    #ll+= x_recon[i][j].log_prob(x[i]).mean(1, keepdims=True).mean(0) 
        return ll

    def sample_from_normal(self, normal):
        return normal.loc

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn['x_recon']
        mu = fwd_rtn['mu']
        logvar = fwd_rtn['logvar']

        kl = self.calc_kl(self, mu, logvar)
        recon = self.calc_ll(self, x, x_recon)

        total = kl - recon
        losses = {'total': total,
                'kl': kl,
                'll': recon}
        return losses


__all__ = [
    'VAE'
]