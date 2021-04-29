import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from src.utils.kl_utils import compute_logvar
from src.utils.datasets import MyDataset
import numpy as np
#from torch_geometric.nn import GCNConv
class Encoder(nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_layer_dims, 
                variational=True, 
                non_linear=False, 
                bias=True, 
                sparse=False, 
                log_alpha=None):
        super().__init__()

        self.input_size = input_dim
        self.hidden_dims = hidden_layer_dims
        self.z_dim = hidden_layer_dims[-1]
        self.variational = variational
        self.non_linear = non_linear
        self.layer_sizes_encoder = [input_dim] + hidden_layer_dims
        self.sparse = sparse
        lin_layers = [nn.Linear(dim0, dim1, bias=bias) for dim0, dim1 in zip(self.layer_sizes_encoder[:-1], self.layer_sizes_encoder[1:])]
        
        self.encoder_layers = nn.Sequential(*lin_layers)
        if self.variational:
            self.enc_mean_layer = nn.Linear(self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=bias)
            if not self.sparse:
                self.enc_logvar_layer = nn.Linear(self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=bias)
            else:
                self.log_alpha = log_alpha

    def forward(self, x):
        h1 = x
        for it_layer, layer in enumerate(self.encoder_layers[:-1]):
            h1 = layer(h1)
            if self.non_linear:
                h1 = F.relu(h1)
        if self.variational:
            if not self.sparse:
                mu = self.enc_mean_layer(h1)
                logvar = self.enc_logvar_layer(h1)
            else:
                mu = self.enc_mean_layer(h1)
                logvar = compute_logvar(mu, self.log_alpha)             
            return mu, logvar
        else:
            h1 = self.encoder_layers[-1](h1)
            return h1

class Decoder(nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_layer_dims,
                variational=True, 
                non_linear=False, 
                init_logvar=-3,
                bias=True):
        super().__init__()

        self.input_size = input_dim
        self.hidden_dims = hidden_layer_dims
        self.non_linear = non_linear
        self.variational = variational
        self.init_logvar = init_logvar
        self.layer_sizes_decoder = hidden_layer_dims[::-1] + [input_dim]
        lin_layers = [nn.Linear(dim0, dim1, bias=bias) for dim0, dim1 in zip(self.layer_sizes_decoder[:-1], self.layer_sizes_decoder[1:])]
        self.decoder_layers = nn.Sequential(*lin_layers)
        if self.variational:
            tmp_noise_par = torch.FloatTensor(1, self.input_size).fill_(self.init_logvar)
            self.dec_logvar = torch.nn.Parameter(data=tmp_noise_par, requires_grad=True)
            del tmp_noise_par

    def forward(self, z):
        x_rec = z
        for it_layer, layer in enumerate(self.decoder_layers):
            x_rec = layer(x_rec)
            if self.non_linear:
                x_rec = F.relu(x_rec)
        if self.variational:
            x_rec = Normal(loc=x_rec, scale=self.dec_logvar.exp().pow(0.5))
        else:
            x_rec = self.decoder_layers[-1](x_rec)

        return x_rec


class GCN(nn.Module):
    def __init__(
                self, 
                input_dim,
                classes):
        super().__init__()

        self.input_size = input_dim
        self.classes = classes
        self.conv1 = GCNConv(self.input_size, 16)
        self.conv2 = GCNConv(16, 16)
        self.classifier = nn.Linear(16, self.classes)

    def forward(self, z, edge_index, edge_attr):
        x = self.conv1(z, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

