#Code from: https://github.com/thomassutter/MoPoE

import torch
import torch.nn as nn
import hydra 

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)
        #return x.view(x.size(1), *self.ndims) #QUICK FIX FOR MMVAE FOR NOW


class VariationalEncoder(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, 
        z_dim,
        **kwargs):
        super().__init__()

        self.shared_encoder = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),     # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # -> (128, 4, 4)
            nn.ReLU(),
            Flatten(),                                                # -> (2048)
            nn.Linear(2048, z_dim),       # -> (latent_dim)
            nn.ReLU(),
        )

        self.class_mu = nn.Linear(z_dim, z_dim)
        self.class_logvar = nn.Linear(z_dim, z_dim)

    def forward(self, x):
        h = self.shared_encoder(x)
        return self.class_mu(h), self.class_logvar(h)

class Decoder(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, 
        z_dim,
        dec_dist,
        **kwargs):
        super().__init__()
        self.dec_dist = dec_dist
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 2048),                                # -> (2048)
            nn.ReLU(),
            Unflatten((128, 4, 4)),                                                            # -> (128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),                   # -> (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (3, 28, 28)
        )

    def forward(self, z):
        x_hat = torch.empty(z.shape[0], z.shape[1], 3, 28, 28).to(z.device)
        if self.training:
            for sample in range(0,z.shape[0]):
                x_hat_ = self.decoder(z[sample, :, :])
                x_hat[sample, :, :, :, :] = x_hat_
            x_hat = hydra.utils.instantiate(self.dec_dist, x=x_hat)
        else:
            x_hat = self.decoder(z)
            x_hat = hydra.utils.instantiate(self.dec_dist, x=x_hat)
        # x_hat = torch.sigmoid(x_hat)
        return x_hat