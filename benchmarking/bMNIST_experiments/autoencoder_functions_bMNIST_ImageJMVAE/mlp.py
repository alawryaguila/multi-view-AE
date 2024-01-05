

import torch
import torch.nn as nn
import hydra 

class VariationalEncoder(nn.Module):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, 
        z_dim,
        **kwargs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
        )
        self.mu_layer = torch.nn.Linear(512, z_dim)
        self.var_layer = torch.nn.Sequential(
            torch.nn.Linear(512, z_dim), torch.nn.Softplus()
        )

    def forward(self, x):
        h = self.layers(x)
        mu = self.mu_layer(h)
        logvar = torch.log(self.var_layer(h))

        return mu, logvar

class Decoder(nn.Module):
    """Parametrizes p(x|z).
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, 
        z_dim,
        dec_dist,
        **kwargs):
        super().__init__()
        self.dec_dist = dec_dist
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(z_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 784),
        )
    def forward(self, z):
        x = self.layers(z)
        return hydra.utils.instantiate(self.dec_dist, x=x)
    