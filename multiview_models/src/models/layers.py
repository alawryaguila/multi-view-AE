import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_layer_dims, variational=True, non_linear=False, bias=True):
        super().__init__()

        self.input_size = input_dim
        self.hidden_dims = hidden_layer_dims
        self.variational = variational
        self.non_linear = non_linear
        self.layer_sizes_encoder = [input_dim] + hidden_layer_dims
        lin_layers = [nn.Linear(dim0, dim1, bias=bias) for dim0, dim1 in zip(self.layer_sizes_encoder[:-1], self.layer_sizes_encoder[1:])]
        
        self.encoder_layers = nn.Sequential(*lin_layers)
        if self.variational:
            self.mean_layer = nn.Linear(self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=bias)
            self.logvar_layer = nn.Linear(self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=bias)

    def forward(self, x):
        h1 = x
        for it_layer, layer in enumerate(self.encoder_layers[:-1]):
            h1 = layer(h1)
            if self.non_linear:
                h1 = F.relu(h1)
        if self.variational:
            mu = self.mean_layer(h1)
            logvar = self.logvar_layer(h1)
            return mu, logvar
        else:
            h1 = self.encoder_layers[-1](h1)
            return h1

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_layer_dims, non_linear=False, bias=True):
        super().__init__()

        self.input_size = input_dim
        self.hidden_dims = hidden_layer_dims
        self.non_linear = non_linear

        layer_sizes_decoder = hidden_layer_dims[::-1] + [input_dim]
        lin_layers = [nn.Linear(dim0, dim1, bias=bias) for dim0, dim1 in zip(layer_sizes_decoder[:-1], layer_sizes_decoder[1:])]
        self.decoder_layers = nn.Sequential(*lin_layers)

    def forward(self, z):
        x_rec = z
        for it_layer, layer in enumerate(self.decoder_layers):
            x_rec = layer(x_rec)
            if it_layer < len(self.decoder_layers) -1:
                if self.non_linear:
                    x_rec = F.relu(x_rec)

        return x_rec


