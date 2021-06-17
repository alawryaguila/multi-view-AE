import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .layers import Encoder, Decoder 
from .utils_deep import Optimisation_VAE
import numpy as np
from ..utils.kl_utils import compute_logvar, compute_kl, compute_kl_sparse
from ..utils.calc_utils import ProductOfExperts, MeanRepresentation
class VAE(nn.Module, Optimisation_VAE):
    
    def __init__(
                self, 
                input_dims, 
                config):

        ''' 
        Initialise Variational Autoencoder model.

        input_dims: The input data dimension.
        config: Configuration dictionary.

        '''

        super().__init__()
        self._config = config
        self.model_type = 'joint_VAE'
        self.input_dims = input_dims
        hidden_layer_dims = config['hidden_layers'].copy()
        self.z_dim = config['latent_size']
        hidden_layer_dims.append(self.z_dim)
        self.non_linear = config['non_linear']
        self.beta = config['beta']
        self.learning_rate = config['learning_rate']
        self.SNP_model = config['SNP_model']
        self.joint_representation = True
        self.join_type = 'Mean'
        if self.join_type == 'PoE':
            self.join_z = ProductOfExperts()
        elif self.join_type == 'Mean':
            self.join_z = MeanRepresentation()
        else:
            print("Incorrect join method")
            exit()
        self.threshold = config['dropout_threshold']
        if self.threshold!=0:
            self.sparse = True
            self.model_type = 'joint_sparse_VAE'
            self.log_alpha = torch.nn.Parameter(torch.FloatTensor(1, self.z_dim).normal_(0,0.01))
        else:
            self.log_alpha = None
            self.sparse = False
        self.n_views = len(input_dims)
        self.encoders = torch.nn.ModuleList([Encoder(input_dim=input_dim, hidden_layer_dims=hidden_layer_dims, variational=True, non_linear=self.non_linear, sparse=self.sparse, log_alpha=self.log_alpha) for input_dim in self.input_dims])
        self.decoders = torch.nn.ModuleList([Decoder(input_dim=input_dim, hidden_layer_dims=hidden_layer_dims, variational=True, non_linear=self.non_linear) for input_dim in self.input_dims])
        self.optimizers = [torch.optim.Adam(list(self.encoders[i].parameters()) + list(self.decoders[i].parameters()),
                                      lr=self.learning_rate) for i in range(self.n_views)]
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
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

    def decode(self, z):
        x_recon = []
        for i in range(self.n_views):
            mu_out = self.decoders[i](z)
            x_recon.append(mu_out)
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
        dropout = self.dropout()
        keep = (dropout < self.threshold).squeeze().cpu()
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
        if self.sparse:
            kl+= compute_kl_sparse(mu, logvar)
        else:
            kl+= compute_kl(mu, logvar)
        return self.beta*kl

    @staticmethod
    def calc_ll(self, x, x_recon):
        ll = 0
        for i in range(self.n_views):
            ll+= torch.mean(x_recon[i].log_prob(x[i]).sum(dim=1))
        return ll


    @staticmethod
    def recon_loss(self, x, x_recon):
        recon_loss = 0   
        for i in range(self.n_views):
            recon_loss+= torch.mean(((x_recon[i] - x[i])**2).sum(dim=1))
        return recon_loss


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