

import torch
import torch.nn as nn
import hydra 

class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""

    def forward(self, x):
        return x * torch.sigmoid(x)

class VariationalEncoder(nn.Module):
    """Parametrizes q(z|y).
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, 
        z_dim,
        **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(10, 512) #supposed to be nn.Embedding(10, 512) but not sure how to get this to work
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, z_dim)
        self.fc32 = nn.Linear(512, z_dim)
        self.swish = Swish()

    def forward(self, x):
        h_1 = self.swish(self.fc1(x))
        h_2 = self.swish(self.fc2(h_1))
        mu = self.fc31(h_2)
        logvar = self.fc32(h_2)
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
        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 10)
        self.swish = Swish()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        x = self.fc4(h)
        return hydra.utils.instantiate(self.dec_dist, x=x)  # no softmax here