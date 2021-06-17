import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .layers import Encoder, Decoder, Classifier
from .utils_deep import Optimisation_VAE
import numpy as np
from ..utils.kl_utils import compute_logvar, compute_kl, compute_kl_sparse

class VAE_classifier(nn.Module, Optimisation_VAE):
    
    def __init__(
                self, 
                input_dims, 
                config,
                n_labels,
                initial_weights=None):

        ''' 
        Initialise Variational Autoencoder model with classifier for classifying labels from latent space.

        input_dims: The input data dimension.
        config: Configuration dictionary.

        '''

        super().__init__()
        self._config = config
        self.model_type = 'VAE_classifier'
        self.input_dims = input_dims
        hidden_layer_dims = config['hidden_layers'].copy()
        self.z_dim = config['latent_size']
        self.n_labels = n_labels
        hidden_layer_dims.append(self.z_dim)
        self.non_linear = config['non_linear']
        self.beta = config['beta']
        self.learning_rate = config['learning_rate']
        self.threshold = config['dropout_threshold']
        self.SNP_model = config['SNP_model']
        self.joint_representation = False
        if self.threshold!=0:
            self.sparse = True
            self.model_type = 'sparse_VAE_classifier'
            self.log_alpha = torch.nn.Parameter(torch.FloatTensor(1, self.z_dim).normal_(0,0.01))
        else:
            self.log_alpha = None
            self.sparse = False
        self.n_views = len(input_dims)
        self.encoders = torch.nn.ModuleList([Encoder(input_dim=input_dim, hidden_layer_dims=hidden_layer_dims, variational=True, non_linear=self.non_linear, sparse=self.sparse, log_alpha=self.log_alpha) for input_dim in self.input_dims])
        self.decoders = torch.nn.ModuleList([Decoder(input_dim=input_dim, hidden_layer_dims=hidden_layer_dims, variational=True, non_linear=self.non_linear) for input_dim in self.input_dims])
        self.classifiers = torch.nn.ModuleList([Classifier(input_dim=self.z_dim, hidden_layer_dims = self.z_dim, output_dim=self.n_labels, non_linear=False) for view in range(self.n_views)])
        
        self.optimizers = [torch.optim.Adam(list(self.encoders[i].parameters()) + list(self.decoders[i].parameters())+list(self.classifiers[i].parameters()),
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

    def classify(self, z):
        pred = []
        for i in range(self.n_views): 
            pred_ = self.classifiers[i](z[i])
            pred.append(pred_)
        return pred


    def decode(self, z):
        x_recon = []
        for i in range(self.n_views):
            temp_recon = [self.decoders[i](z[j]) for j in range(self.n_views)]
            x_recon.append(temp_recon)
            del temp_recon 
        return x_recon

    def forward(self, data):
        x = data[0]
        self.zero_grad()
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        pred = self.classify(z)
        x_recon = self.decode(z)
        fwd_rtn = {'x_recon': x_recon,
                    'mu': mu,
                    'logvar': logvar,
                    'pred': pred}
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
                kl+= -0.5*torch.mean(1 + logvar[i] - mu[i].pow(2) - logvar[i].exp(), dim=1).mean(0)
        return self.beta*kl

    @staticmethod
    def calc_ll(self, x, x_recon):
        ll = 0    
        for i in range(self.n_views):
            for j in range(self.n_views):
                    ll+= x_recon[i][j].log_prob(x[i]).mean(1, keepdims=True).mean(0) 
        return ll


    @staticmethod
    def calc_ce(self, y, pred):
        ce = 0    
        for i in range(self.n_views):
            ce+= F.cross_entropy(pred[i], y)
        return ce

    def sample_from_normal(self, normal):
        return normal.loc

    def loss_function(self, data, fwd_rtn):
        x, y = data[0], data[1]
        x_recon = fwd_rtn['x_recon']
        mu = fwd_rtn['mu']
        logvar = fwd_rtn['logvar']
        pred = fwd_rtn['pred']

        kl = self.calc_kl(self, mu, logvar)
        recon = self.calc_ll(self, x, x_recon)
        ce = self.calc_ce(self, y, pred)
        total = kl + ce - recon
        losses = {'total': total,
                'kl': kl,
                'll': recon,
                'ce': ce}
        return losses


__all__ = [
    'VAE'
]