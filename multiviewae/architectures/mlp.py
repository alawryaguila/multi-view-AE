import ast
import torch
import hydra

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter

class Encoder(nn.Module):
    """MLP Encoder

    Args:
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
        hidden_layer_dim (list): Number of nodes per hidden layer.
        non_linear (bool): Whether to include a ReLU() function between layers.
        bias (bool): Whether to include a bias term in hidden layers.
        enc_dist (multiviewae.base.distributions.Default): Encoder distribution.
    """
    def __init__(
        self,
        input_dim,
        z_dim,
        hidden_layer_dim,
        non_linear,
        bias,
        enc_dist
    ):
        super().__init__()

        self.input_size = input_dim
        self.z_dim = z_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.bias = bias
        self.enc_dist = enc_dist
        self.non_linear = non_linear
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
            if self.non_linear:
                h1 = F.relu(h1)
        h1 = self.encoder_layers[-1](h1)

        return h1

class VariationalEncoder(Encoder):
    """Variational MLP Encoder

    Args:
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
        hidden_layer_dim (list): Number of nodes per hidden layer.
        non_linear (bool): Whether to include a ReLU() function between layers.
        bias (bool): Whether to include a bias term in hidden layers.
        sparse (bool): Whether to enforce sparsity of the encoding distribution.
        log_alpha (float): Log of the dropout parameter.
        enc_dist (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoder distribution.
    """
    def __init__(
        self,
        input_dim,
        z_dim,
        hidden_layer_dim,
        non_linear,
        bias,
        sparse,
        log_alpha,
        enc_dist
    ):
        super().__init__(input_dim=input_dim,
                        z_dim=z_dim,
                        hidden_layer_dim=hidden_layer_dim,
                        bias=bias,
                        non_linear=non_linear,
                        enc_dist=enc_dist)

        self.sparse = sparse
        self.non_linear = non_linear
        self.log_alpha = log_alpha

        self.encoder_layers = self.encoder_layers[:-1]
        self.enc_mean_layer = nn.Linear(
            self.layer_sizes[-2],
            self.layer_sizes[-1],
            bias=self.bias,
        )

        if not self.sparse:
            self.enc_logvar_layer = nn.Linear(
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

class ConditionalVariationalEncoder(Encoder):
    """MLP Variational Conditional Encoder

    Args:
        y (list):  
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
        hidden_layer_dim (list): Number of nodes per hidden layer.
        non_linear (bool): Whether to include a ReLU() function between layers.
        bias (bool): Whether to include a bias term in hidden layers.
        sparse (bool): Whether to enforce sparsity of the encoding distribution.
        log_alpha (float): Log of the dropout parameter.
        enc_dist (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoder distribution.
        num_cat (int): Number of categories of the labels.
    """
    def __init__(
        self,
        input_dim,
        z_dim,
        hidden_layer_dim,
        non_linear,
        bias,
        sparse,
        log_alpha,
        enc_dist,
        num_cat
    ):
        super().__init__(input_dim=input_dim,
                        z_dim=z_dim,
                        hidden_layer_dim=hidden_layer_dim,
                        bias=bias,
                        non_linear=non_linear,
                        enc_dist=enc_dist)

        self.num_cat = num_cat
        self.sparse = sparse
        self.non_linear = non_linear
        self.log_alpha = log_alpha

        self.layer_sizes = [input_dim + num_cat] + self.hidden_layer_dim + [z_dim]
        lin_layers = [
            nn.Linear(dim0, dim1, bias=self.bias)
            for dim0, dim1 in zip(
                self.layer_sizes[:-1], self.layer_sizes[1:]
            )
        ]
        self.encoder_layers = nn.Sequential(*lin_layers)

        self.encoder_layers = self.encoder_layers[:-1]
        self.enc_mean_layer = nn.Linear(
            self.layer_sizes[-2],
            self.layer_sizes[-1],
            bias=self.bias,
        )

        if not self.sparse:
            self.enc_logvar_layer = nn.Linear(
                self.layer_sizes[-2],
                self.layer_sizes[-1],
                bias=self.bias,
            )

    def set_labels(self, labels):
        self.labels = labels 

    def forward(self, x):
        c = F.one_hot(self.labels, self.num_cat)
        x_cond = torch.hstack((x, c))    

        h1 = x_cond
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
    
class Decoder(nn.Module):
    """MLP Decoder
    
    Args:
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
        hidden_layer_dim (list): Number of nodes per hidden layer. The layer order is reversed e.g. [100, 50, 5] becomes [5, 50, 100].
        non_linear (bool): Whether to include a ReLU() function between layers.
        bias (bool): Whether to include a bias term in hidden layers.
        dec_dist (multiviewae.base.distributions.Default, multiviewae.base.distributions.Bernoulli): Decoder distribution.
        init_logvar (int, float): Initial value for log variance of decoder. Unused in Decoder class.
    """
    def __init__(
        self,
        input_dim,
        z_dim,
        hidden_layer_dim,
        non_linear,
        bias,
        dec_dist,
        init_logvar=None
    ):
        super().__init__()

        self.input_size = input_dim
        self.z_dim = z_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.bias = bias
        self.dec_dist = dec_dist
        self.non_linear = non_linear
        self.layer_sizes = [z_dim] + self.hidden_layer_dim + [input_dim]
        lin_layers = [
            nn.Linear(dim0, dim1, bias=self.bias)
            for dim0, dim1 in zip(
                self.layer_sizes[:-1], self.layer_sizes[1:]
            )
        ]

        self.decoder_layers = nn.Sequential(*lin_layers)

    def forward(self, z):
        x_rec = z
        for it_layer, layer in enumerate(self.decoder_layers[:-1]):
            x_rec = layer(x_rec)
            if self.non_linear:
                x_rec = F.relu(x_rec)
        x_rec = self.decoder_layers[-1](x_rec)
        x_rec = hydra.utils.instantiate(self.dec_dist, x=x_rec)
        return x_rec

class VariationalDecoder(Decoder):
    """MLP Variational Decoder

    Args:
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
        hidden_layer_dim (list): Number of nodes per hidden layer. The layer order is reversed e.g. [100, 50, 5] becomes [5, 50, 100].
        non_linear (bool): Whether to include a ReLU() function between layers.
        bias (bool): Whether to include a bias term in hidden layers.
        init_logvar (int, float): Initial value for log variance of decoder.
        dec_dist (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoder distribution.
    """
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
                non_linear=non_linear,
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
    
class ConditionalVariationalDecoder(Decoder):
    """MLP Conditinal Variational Decoder

    Args:
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
        hidden_layer_dim (list): Number of nodes per hidden layer. The layer order is reversed e.g. [100, 50, 5] becomes [5, 50, 100].
        non_linear (bool): Whether to include a ReLU() function between layers.
        bias (bool): Whether to include a bias term in hidden layers.
        init_logvar (int, float): Initial value for log variance of decoder.
        dec_dist (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoder distribution.
        num_cat (int): Number of categories of the labels.
    """
    def __init__(
        self,
        input_dim,
        z_dim,
        hidden_layer_dim,
        bias,
        non_linear,
        init_logvar,
        dec_dist, 
        num_cat
    ):
        super().__init__(input_dim=input_dim,
                z_dim=z_dim,
                hidden_layer_dim=hidden_layer_dim,
                non_linear=non_linear,
                bias=bias, dec_dist=dec_dist)

        self.num_cat = num_cat
        self.non_linear = non_linear
        self.init_logvar = init_logvar
        self.dec_dist = dec_dist

        self.layer_sizes = [z_dim + num_cat] + self.hidden_layer_dim + [input_dim]
        lin_layers = [
            nn.Linear(dim0, dim1, bias=self.bias)
            for dim0, dim1 in zip(
                self.layer_sizes[:-1], self.layer_sizes[1:]
            )
        ]

        self.decoder_layers = nn.Sequential(*lin_layers)

        self.decoder_layers = self.decoder_layers[:-1]
        self.dec_mean_layer = nn.Linear(
            self.layer_sizes[-2],
            self.layer_sizes[-1],
            bias=self.bias,
        )
        tmp_noise_par = torch.FloatTensor(1, self.input_size).fill_(
            self.init_logvar
        )
        self.logvar_out = Parameter(data=tmp_noise_par, requires_grad=True)

    def set_labels(self, labels): 
        self.labels = labels 

    def forward(self, z):
        c = F.one_hot(self.labels, self.num_cat)
        if (len(z.size()) == 3 and len(c.size()) == 2): # NOTE: for mmvae which uses rsample() instead of sample()
            z_cond = torch.cat((z, c.repeat(z.size()[0],1,1)), dim=2)
        else:
            z_cond = torch.hstack((z, c))
        
        x_rec = z_cond
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
    """MLP Discriminator
    
    Args:
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of output dimensions.
        hidden_layer_dim (list): Number of nodes per hidden layer.
        non_linear (bool): Whether to include a ReLU() function between layers.
        bias (bool): Whether to include a bias term in hidden layers.
        dropout_threshold (float): Dropout threshold of layers.
        is_wasserstein (bool): Whether model employs a wasserstein loss.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_dim,
        non_linear,
        bias,
        dropout_threshold,
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
