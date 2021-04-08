import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .layers import Encoder, Decoder, GCN 
from .utils_deep import Optimisation_GVCCA
import numpy as np
from utils.kl_utils import compute_logvar, compute_kl, compute_kl_sparse

from torch_geometric.utils import dense_to_sparse

class GVCCA(nn.Module, Optimisation_GVCCA):
    
    def __init__(
                self, 
                input_dims, 
                config,
                classes,
                lam=0.99
                ):

        ''' 
        Initialise variational graph model model.

        input_dims: The input data dimension.
        config: Configuration dictionary.

        '''

        super().__init__()
        self._config = config
        self.model_type = 'GVCCA'
        self.input_dims = input_dims
        self.hidden_layer_dims = config['hidden_layers']
        self.z_dim = config['latent_size']
        self.hidden_layer_dims.append(self.z_dim)
        self.non_linear = config['non_linear']
        self.beta = config['beta']
        self.learning_rate = config['learning_rate']
        self.classes = classes
        self.sparse = config['sparse']        
        if self.sparse:
            self.threshold = config['dropout_threshold']
            self.log_alpha = torch.nn.Parameter(torch.FloatTensor(1, self.z_dim).normal_(0,0.01))
        else:
            self.log_alpha = None
        self.n_views = len(input_dims)
        self.encoders = torch.nn.ModuleList([Encoder(input_dim=input_dim, hidden_layer_dims=self.hidden_layer_dims, variational=True, non_linear=self.non_linear, sparse=False, log_alpha=self.log_alpha) for input_dim in self.input_dims])
        self.gcn = GCN(input_dim = self.z_dim, classes = self.classes)
        encoder_optimisers = [torch.optim.Adam(list(self.encoders[i].parameters()),lr=self.learning_rate) for i in range(self.n_views)]
        gcn_optimiser = [torch.optim.Adam(list(self.gcn.parameters()), lr=self.learning_rate)]
        self.optimizers = encoder_optimisers + gcn_optimiser 
        self.t = torch.nn.Parameter(torch.rand(1))
        self.theta = torch.nn.Parameter(torch.rand(1))
        self.lam = lam

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

    def forward(self, x):
        self.zero_grad()
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        pred = self.encode_gcn(z)
        fwd_rtn = {'pred': pred,
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

    def emb2adj(self, z):
        pairwise_distances = [embedding @ embedding.T for embedding in z]
        # Take the mean distance
        pairwise_distances = torch.stack(pairwise_distances, dim=0).mean(dim=0)
        A = 1 / (1 + torch.exp(-self.t * (pairwise_distances + self.theta)))
        edge_index, edge_attr = dense_to_sparse(A)
        return edge_index, edge_attr

    def encode_gcn(self, embeddings):
        # sum the embeddings to give joint representation (Bxlatent)
        #joint_representation = torch.cat(embeddings, axis=1)
        joint_representation = torch.mean(torch.stack(embeddings), axis=0)
        # Calculate adjacency matrix (BxB)
        edge_index, edge_attr = self.emb2adj(embeddings)
        # graph convolution
        pred = self.gcn(joint_representation, edge_index, edge_attr)
        return pred

    @staticmethod
    def calc_kl(self, mu, logvar):
        '''
        VAE: Implementation from: https://arxiv.org/abs/1312.6114


        '''
        kl = 0
        for i in range(self.n_views):
            kl+= -0.5*torch.sum(1 + logvar[i] - mu[i].pow(2) - logvar[i].exp(), dim=1).mean(0)

        return self.beta*kl/self.n_views

    def loss_function(self, labels, fwd_rtn):
        pred = fwd_rtn['pred']
        mu = fwd_rtn['mu']
        logvar = fwd_rtn['logvar']

        kl = self.calc_kl(self, mu, logvar)
        #labels = labels.to(dtype=torch.long)
        pred_loss = F.nll_loss(pred, labels)
        total = self.lam*pred_loss + (1-self.lam)*kl
        losses = {'total': total,
                'kl': kl,
                'prediction': pred_loss}
        return losses


__all__ = [
    'GVCCA'
]