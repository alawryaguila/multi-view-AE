import ast
import torch
import hydra

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter

# TODO: convert this to LightningModule
# TODO: change name to MLPEncoder?
class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        z_dim,
        hidden_layer_dim,
        bias
    ):
        super().__init__()

        self.input_size = input_dim
        self.z_dim = z_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.bias = bias

        self.layer_sizes = [input_dim] + self.hidden_layer_dim + [z_dim]
        lin_layers = [
            nn.Linear(dim0, dim1, bias=self.bias)
            for dim0, dim1 in zip(
                self.layer_sizes[:-1], self.layer_sizes[1:]
            )
        ]

        self.encoder_layers = nn.Sequential(*lin_layers)

    def forward(self, x):
        h1 = x
        for it_layer, layer in enumerate(self.encoder_layers[0:-1]):
            h1 = layer(h1)
        h1 = self.encoder_layers[-1](h1)
        return h1

class VariationalEncoder(Encoder):
    def __init__(
        self,
        input_dim,
        z_dim,
        hidden_layer_dim,
        bias,
        sparse,     #TODO: only variational can be sparse?
        non_linear,
        log_alpha,
        enc_dist,
    ):
        super().__init__(input_dim=input_dim,
                        z_dim=z_dim,
                        hidden_layer_dim=hidden_layer_dim,
                        bias=bias)

        self.sparse = sparse
        self.non_linear = non_linear
        self.log_alpha = log_alpha
        self.enc_dist = enc_dist

        self.encoder_layers = self.encoder_layers[:-1]
        self.enc_mean_layer = nn.Linear(
            self.layer_sizes[-2],
            self.layer_sizes[-1],
            bias=self.bias,
        )

        if not self.sparse:
            self.enc_logvar_layer = nn.Linear(  # TODO: this is not set for sparse=True?
                self.layer_sizes[-2],
                self.layer_sizes[-1],
                bias=self.bias,
            )

    def forward(self, x):
        h1 = x
        for it_layer, layer in enumerate(self.encoder_layers):
            h1 = layer(h1)
            if self.non_linear:
                h1 = F.relu(h1)

        if not self.sparse:
            mu = self.enc_mean_layer(h1)
            logvar = self.enc_logvar_layer(h1)
        else:
            mu = self.enc_mean_layer(h1)
            logvar = self.log_alpha + 2 * torch.log(torch.abs(mu) + 1e-8)

        return mu, logvar


# TODO: convert this to LightningModule
class Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        z_dim,
        hidden_layer_dim,
        bias,
        dec_dist
    ):
        super().__init__()

        self.input_size = input_dim
        self.z_dim = z_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.bias = bias
        self.dec_dist = dec_dist

        self.layer_sizes = [z_dim] + self.hidden_layer_dim[::-1] + [input_dim]
        lin_layers = [
            nn.Linear(dim0, dim1, bias=self.bias)
            for dim0, dim1 in zip(
                self.layer_sizes[:-1], self.layer_sizes[1:]
            )
        ]

        self.decoder_layers = nn.Sequential(*lin_layers)

    def forward(self, z):
        x_rec = z
        for it_layer, layer in enumerate(self.decoder_layers):
            x_rec = layer(x_rec)
        x_rec = hydra.utils.instantiate(self.dec_dist, x_rec)
        return x_rec

class VariationalDecoder(Decoder):
    def __init__(
        self,
        input_dim,
        z_dim,
        hidden_layer_dim,
        bias,
        non_linear,
        init_logvar,
        dec_dist
    ):
        super().__init__(input_dim=input_dim,
                z_dim=z_dim,
                hidden_layer_dim=hidden_layer_dim,
                bias=bias, dec_dist=dec_dist)

        self.non_linear = non_linear
        self.init_logvar = init_logvar
        self.dec_dist = dec_dist

        self.decoder_layers = self.decoder_layers[:-1]
        self.dec_mean_layer = nn.Linear(
            self.layer_sizes[-2],
            self.layer_sizes[-1],
            bias=self.bias,
        )
        tmp_noise_par = torch.FloatTensor(1, self.input_size).fill_(
            self.init_logvar
        )
        tmp_noise_par = torch.FloatTensor(1, input_dim).fill_(init_logvar)
        self.logvar_out = Parameter(data=tmp_noise_par, requires_grad=True)

    def forward(self, z):
        x_rec = z
        for it_layer, layer in enumerate(self.decoder_layers):
            x_rec = layer(x_rec)
            if self.non_linear:
                x_rec = F.relu(x_rec)

        x_rec = self.dec_mean_layer(x_rec)
        x_rec = hydra.utils.instantiate(
            self.dec_dist, loc=x_rec, scale=torch.exp(0.5 * self.logvar_out)
        )
        return x_rec

class Discriminator(nn.Module):
    def __init__(
        self,
        hidden_layer_dim,
        bias,
        non_linear,
        dropout_threshold,
        input_dim,
        output_dim,
        is_wasserstein
    ):
        super().__init__()
        self.bias = bias
        self.non_linear = non_linear
        self.dropout_threshold = dropout_threshold
        self.is_wasserstein = is_wasserstein

        self.layer_sizes = [input_dim] + hidden_layer_dim + [output_dim]

        lin_layers = [
            nn.Linear(dim0, dim1, bias=self.bias)
            for dim0, dim1 in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        ]
        self.linear_layers = nn.Sequential(*lin_layers)

    def forward(self, x):
        for it_layer, layer in enumerate(self.linear_layers):
            x = F.dropout(layer(x), self.dropout_threshold, training=self.training)
            if it_layer < len(self.linear_layers) - 1:
                if self.non_linear:
                    x = F.relu(x)
            else:
                if self.is_wasserstein:
                    return x
                elif self.layer_sizes[-1] > 1:
                    x = nn.Softmax(dim=-1)(x)
                else:
                    x = torch.sigmoid(x)
        return x
