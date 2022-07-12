import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from ..utils.kl_utils import compute_logvar
from ..utils.datasets import MyDataset
import numpy as np
from torch.nn import Parameter


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layer_dims,
        variational=True,
        non_linear=False,
        bias=True,
        sparse=False,
        log_alpha=None,
    ):
        super().__init__()

        self.input_size = input_dim
        self.hidden_dims = hidden_layer_dims
        self.z_dim = hidden_layer_dims[-1]
        self.variational = variational
        self.non_linear = non_linear
        self.layer_sizes_encoder = [input_dim] + hidden_layer_dims
        self.sparse = sparse
        lin_layers = [
            nn.Linear(dim0, dim1, bias=bias)
            for dim0, dim1 in zip(
                self.layer_sizes_encoder[:-1], self.layer_sizes_encoder[1:]
            )
        ]

        if self.variational:
            self.encoder_layers = nn.Sequential(*lin_layers[0:-1])
            self.enc_mean_layer = nn.Linear(
                self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=bias
            )
            if not self.sparse:
                self.enc_logvar_layer = nn.Linear(
                    self.layer_sizes_encoder[-2],
                    self.layer_sizes_encoder[-1],
                    bias=bias,
                )
            else:
                self.log_alpha = log_alpha
        else:
            self.encoder_layers = nn.Sequential(*lin_layers)

    def forward(self, x):
        h1 = x
        if self.variational:
            for it_layer, layer in enumerate(self.encoder_layers):
                h1 = layer(h1)
                if self.non_linear:
                    h1 = F.relu(h1)

            if not self.sparse:
                mu = self.enc_mean_layer(h1)
                logvar = self.enc_logvar_layer(h1)
            else:
                mu = self.enc_mean_layer(h1)
                logvar = compute_logvar(mu, self.log_alpha)
            return mu, logvar
        else:
            for it_layer, layer in enumerate(self.encoder_layers[0:-1]):
                h1 = layer(h1)
                if self.non_linear:
                    h1 = F.relu(h1)
            h1 = self.encoder_layers[-1](h1)
            return h1

class Discriminator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layer_dims,
        output_dim,
        dropout=0,
        non_linear=True,
        wasserstein=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_layer_dims
        self.dropout = dropout
        self.layer_sizes = [input_dim] + hidden_layer_dims + [output_dim]
        self.non_linear = non_linear
        self.wasserstein = wasserstein
        lin_layers = [
            nn.Linear(dim0, dim1)
            for dim0, dim1 in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]
        self.linear_layers = nn.Sequential(*lin_layers)

    def forward(self, x):
        for it_layer, layer in enumerate(self.linear_layers):
            x = F.dropout(layer(x), self.dropout, training=self.training)
            if it_layer < len(self.linear_layers) - 1:
                if self.non_linear:
                    x = F.relu(x)
            else:
                if self.wasserstein:
                    return x
                elif self.layer_sizes[-1] > 1:
                    x = nn.Softmax(dim=-1)(x)
                else:
                    x = torch.sigmoid(x)
        return x


class Classifier(nn.Module):
    def __init__(
        self, input_dim, hidden_layer_dims, output_dim, non_linear=False, bias=True
    ):
        super().__init__()

        self.input_size = input_dim
        hidden_layer_dims = (
            hidden_layer_dims
            if type(hidden_layer_dims) == list
            else [hidden_layer_dims]
        )
        self.hidden_dims = hidden_layer_dims
        self.output_size = output_dim
        self.non_linear = non_linear
        self.layer_sizes_encoder = [input_dim] + hidden_layer_dims + [output_dim]
        lin_layers = [
            nn.Linear(dim0, dim1, bias=bias)
            for dim0, dim1 in zip(
                self.layer_sizes_encoder[:-1], self.layer_sizes_encoder[1:]
            )
        ]
        self.encoder_layers = nn.Sequential(*lin_layers)

    def forward(self, x):
        h1 = x
        for it_layer, layer in enumerate(self.encoder_layers[:-1]):
            h1 = layer(h1)
            if self.non_linear:
                h1 = F.relu(h1)

        h1 = F.softmax(self.encoder_layers[-1](h1), dim=1)
        return h1


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layer_dims,
        variational=True,
        non_linear=False,
        init_logvar=-3,
        dist=None,
        bias=True,
    ):
        super().__init__()

        self.input_size = input_dim
        self.hidden_dims = hidden_layer_dims
        self.non_linear = non_linear
        self.variational = variational
        self.init_logvar = init_logvar
        self.dist = dist
        self.layer_sizes_decoder = hidden_layer_dims[::-1] + [input_dim]
        lin_layers = [
            nn.Linear(dim0, dim1, bias=bias)
            for dim0, dim1 in zip(
                self.layer_sizes_decoder[:-1], self.layer_sizes_decoder[1:]
            )
        ]

        if self.variational:
            self.decoder_layers = nn.Sequential(*lin_layers[0:-1])
            self.decoder_mean_layer = nn.Linear(
                self.layer_sizes_decoder[-2], self.layer_sizes_decoder[-1], bias=bias
            )
            tmp_noise_par = torch.FloatTensor(1, self.input_size).fill_(
                self.init_logvar
            )
            if self.dist == "gaussian":
                tmp_noise_par = torch.FloatTensor(1, input_dim).fill_(init_logvar)
                self.logvar_out = Parameter(data=tmp_noise_par, requires_grad=True)
        else:
            self.decoder_layers = nn.Sequential(*lin_layers)

    def forward(self, z):
        x_rec = z
        for it_layer, layer in enumerate(self.decoder_layers):
            x_rec = layer(x_rec)
            if self.non_linear:
                x_rec = F.relu(x_rec)
        if self.variational:
            x_rec = self.decoder_mean_layer(x_rec)
            if self.dist == "gaussian":
                return Normal(loc=x_rec, scale=torch.exp(0.5 * self.logvar_out))
            elif self.dist == "bernoulli":
                return torch.sigmoid(x_rec)
            elif self.dist == "MultivariateGaussian":
                return MultivariateNormal(
                    x_rec,
                    torch.broadcast_to(
                        torch.eye(x_rec.size()[1]),
                        (x_rec.size()[0], x_rec.size()[1], x_rec.size()[1]),
                    ),
                )
                # check this works
        return x_rec
