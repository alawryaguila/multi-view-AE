import torch
import torch.nn as nn
import torch.nn.functional as F
#import pytorch_lightning as pl 
from torch.distributions import Normal
from .layers import Encoder, Decoder 
from .utils_deep import Optimisation_VAE
import numpy as np
from ..utils.kl_utils import compute_kl, compute_kl_sparse, compute_ll
from os.path import join
import pytorch_lightning as pl
import math
def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)

class mmVAE(pl.LightningModule, Optimisation_VAE):
    '''
    Multi-view Variational Autoencoder model using Mixture of Experts approach (https://arxiv.org/abs/1911.03393).
    Code is based on: https://github.com/iffsid/mmvae 
    
    '''
    def __init__(
                self, 
                input_dims,
                z_dim=1,
                hidden_layer_dims=[],
                non_linear=False,
                learning_rate=0.002,
                K=20,
                dist='gaussian',
                **kwargs):
        ''' 
        :param input_dims: columns of input data e.g. [M1 , M2] where M1 and M2 are number of the columns for views 1 and 2 respectively
        :param z_dim: number of latent vectors
        :param hidden_layer_dims: dimensions of hidden layers for encoder and decoder networks.
        :param non_linear: non-linearity between hidden layers. If True ReLU is applied between hidden layers of encoder and decoder networks
        :param learning_rate: learning rate of optimisers.
        :param K: Number of samples.
        :param dist: Approximate distribution of data for log likelihood calculation. Either 'gaussian' or 'bernoulli'.
        '''

        super().__init__()
        self.save_hyperparameters()
        self.model_type = 'joint_VAE'
        self.input_dims = input_dims
        hidden_layer_dims = hidden_layer_dims.copy()
        self.z_dim = z_dim
        hidden_layer_dims.append(self.z_dim)
        self.non_linear = non_linear
        self.dist = dist
        self.learning_rate = learning_rate
        self.joint_representation = True
        self.variational = True
        self.log_alpha = None
        self.sparse = False
        self.K = K
        self.__dict__.update(kwargs)
        self.n_views = len(input_dims)
        self.encoders = torch.nn.ModuleList([Encoder(input_dim=input_dim, hidden_layer_dims=hidden_layer_dims, variational=True, non_linear=self.non_linear, sparse=self.sparse, log_alpha=self.log_alpha) for input_dim in self.input_dims])
        self.decoders = torch.nn.ModuleList([Decoder(input_dim=input_dim, hidden_layer_dims=hidden_layer_dims, variational=True, dist=self.dist, non_linear=self.non_linear) for input_dim in self.input_dims])       
    
    def configure_optimizers(self):
        optimizers = [torch.optim.Adam(list(self.encoders[i].parameters()) + list(self.decoders[i].parameters()),
                                      lr=self.learning_rate) for i in range(self.n_views)]
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
            std = torch.exp(0.5*logvar[i])
            eps = torch.randn_like(mu[i])
            z.append(mu[i]+eps*std)
        return z

    def decode(self, z):
        x_recon = []
        for i in range(self.n_views):
            temp_recon = [self.decoders[j](z[i]) for j in range(self.n_views)] #NOTE: this is other way around to other multiautoencoder models
            x_recon.append(temp_recon)
            del temp_recon 
        return x_recon

    def forward(self, x):
        
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z)
        fwd_rtn = {'x_recon': x_recon,
                    'z': z}
        return fwd_rtn
    
    def forward_k(self, x):
        qz_xs,  zss = [], []
        mu, logvar = self.encode(x)
        for i in range(self.n_views):
            qz_x = Normal(loc=mu[i], scale=logvar[i].exp().pow(0.5)) 
            zs = qz_x.rsample(torch.Size([self.K]))
            qz_xs.append(qz_x)
            zss.append(zs)
        px_zs = self.decode(zss)
        return {'qz_xs':qz_xs, 'px_zs':px_zs, 'zss':zss}

    def moe_iwae(self, x, qz_xs, px_zs, zss): 
        lws = []
        for r, qz_x in enumerate(qz_xs): 
            #print('zss[r]:',zss[r].shape)
            #print('qz_x:',qz_x.loc.shape)
            #print('px_zs[r]:',len(px_zs[r]))
            lpz = Normal(loc=0,scale=1).log_prob(zss[r]).sum(-1)
            #print(lpz.shape)
            #print('px_z batch shape:',px_zs[0][0].loc.view(*px_zs[0][0].batch_shape[:2], -1).shape)
            #print('px_z batch shape:',px_zs[1][0].loc.view(*px_zs[1][0].batch_shape[:2], -1).shape)
            #print('px_z logprob(x[d]) sum: ', px_zs[0][0].log_prob(x[0]).view(*px_zs[0][0].batch_shape[:2], -1).sum(-1).shape)
            #print(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]).shape)
            #print(qz_x.log_prob(zss[r]).sum(-1).shape)
            #print(qz_x.log_prob(zss[r]).shape)
            #print(len(qz_xs))
            lqz_x = self.log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs])) #summing over M modalities for each z to create q(z|x1:M)
            #print(lqz_x.shape)
            lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1).sum(-1)
                    for d, px_z in enumerate(px_zs[r])] #summing over each decoder 
            #print('lpx_z:',lpx_z[0].shape)
            lpx_z = torch.stack(lpx_z).sum(0)
            #print('lpz:',lpz.shape)
            #print('lqz_x:',lqz_x.shape)
            #print('lpx_z:',lpx_z.shape)
            lw = lpz + lpx_z - lqz_x
            #print(lw.shape)
            lws.append(lw)
        return self.log_mean_exp(torch.stack(lws), dim=1).mean(0).sum() #looser iwae bound where have 
    
    def sample_from_normal(self, normal):
        return normal.loc

    def log_mean_exp(self, value, dim=0, keepdim=False):
        return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))

    def loss_function(self, x, fwd_rtn):
        qz_xs, px_zs, zss = fwd_rtn['qz_xs'], fwd_rtn['px_zs'], fwd_rtn['zss']
        total = - self.moe_iwae(x, qz_xs, px_zs, zss)
        losses = {'total': total}
        return losses

    def training_step(self, batch, batch_idx, optimizer_idx):
        fwd_return = self.forward_k(batch)
        loss = self.loss_function(batch, fwd_return)
        self.log(f'train_loss', loss['total'], on_epoch=True, prog_bar=True, logger=True)
        return loss['total']

    def validation_step(self, batch, batch_idx):
        fwd_return = self.forward_k(batch)
        loss = self.loss_function(batch, fwd_return)
        self.log(f'val_loss', loss['total'], on_epoch=True, prog_bar=True, logger=True)
        return loss['total']
    
    def on_train_end(self):
        self.trainer.save_checkpoint(join(self.output_path, 'model.ckpt'))

