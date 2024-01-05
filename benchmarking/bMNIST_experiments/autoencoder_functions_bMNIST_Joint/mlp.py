

import torch
import torch.nn as nn
import hydra 


class VariationalEncoder(nn.Module):
    """Parametrizes q(z|y).
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, 
        z_dim,
        input_dim,
        **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.head_image = torch.nn.Sequential(
            torch.nn.Linear(784, 512), torch.nn.ReLU()
        )

        self.head_labels = torch.nn.Sequential(
            torch.nn.Linear(10, 512), torch.nn.ReLU()
        )

        self.shared_layer = torch.nn.Sequential(
            torch.nn.Linear(512 * 2, 512), torch.nn.ReLU()
        )

        self.mu_layer = torch.nn.Linear(512, z_dim)
        self.var_layer = torch.nn.Sequential(
            torch.nn.Linear(512, z_dim), torch.nn.Softplus()
        )

    def forward(self, x):
        #use input_dim to split x into image and labels
        #check dimension of input_dim
        x_1 = x[:, :784]
        x_2 = x[:, 784:]
        h1 = self.head_image(x_1)
        h2 = self.head_labels(x_2)
        h = torch.cat((h1, h2), dim=-1)
        h = self.shared_layer(h)
        mu = self.mu_layer(h)
        logvar = torch.log(self.var_layer(h))
        return mu, logvar
