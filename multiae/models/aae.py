import torch
from .layers import Encoder, Decoder, Discriminator
from ..base.base_model import BaseModelAAE
import numpy as np
from ..utils.calc_utils import compute_mse, update_dict
from torch.autograd import Variable
import hydra 

class AAE(BaseModelAAE):
    """
    Multi-view Adversarial Autoencoder model with a separate latent representation for each view.

    """

    def __init__(
        self,
        input_dims,
        expt='AAE',
        **kwargs,
    ):

        super().__init__(expt=expt)

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.__dict__.update(self.cfg.model)
        self.__dict__.update(kwargs)
        

        self.cfg.encoder = update_dict(self.cfg.encoder, kwargs)
        self.cfg.decoder = update_dict(self.cfg.decoder, kwargs)

        self.input_dims = input_dims
        self.n_views = len(input_dims)
        
        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(self.cfg.encoder,
                    _recursive_=False,
                    input_dim=input_dim,
                    z_dim=self.z_dim,
                )
                for input_dim in self.input_dims
            ]
        )
        self.decoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(self.cfg.decoder,
                    _recursive_=False,
                    input_dim=input_dim,
                    z_dim=self.z_dim,
                )
                for input_dim in self.input_dims
            ]
        )
        self.discriminator =  hydra.utils.instantiate(self.cfg.discriminator,
                    _recursive_=False,
                    input_dim=self.z_dim,
                    output_dim=(self.n_views + 1),
                )
                
        # TO DO - check discriminator dimensionality is correct (nviews + 1?) and for joint model too

    def configure_optimizers(self):
        optimizers = []
        [
            optimizers.append(
                torch.optim.Adam(
                    list(self.encoders[i].parameters()), lr=self.learning_rate
                )
            )
            for i in range(self.n_views)
        ]
        [
            optimizers.append(
                torch.optim.Adam(
                    list(self.decoders[i].parameters()), lr=self.learning_rate
                )
            )
            for i in range(self.n_views)
        ]
        [
            optimizers.append(
                torch.optim.Adam(
                    list(self.encoders[i].parameters()), lr=self.learning_rate
                )
            )
            for i in range(self.n_views)
        ]
        optimizers.append(
            torch.optim.Adam(
                list(self.discriminator.parameters()), lr=self.learning_rate
            )
        )
        return optimizers

    def encode(self, x):
        z = []
        for i in range(self.n_views):
            z_ = self.encoders[i](x[i])
            z.append(z_)
        return z

    def decode(self, z):
        x_recon = []
        for i in range(self.n_views):
            temp_recon = [self.decoders[i](z[j]) for j in range(self.n_views)]
            x_recon.append(temp_recon)
            del temp_recon
        return x_recon

    def disc(self, z):
        z_real = Variable(torch.randn(z[0].size()[0], self.z_dim) * 1.0).to(self.device)
        d_real = self.discriminator(z_real)
        d_fake = []
        for i in range(self.n_views):
            d = self.discriminator(z[i])
            d_fake.append(d)
        return d_real, d_fake

    def forward_recon(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        fwd_rtn = {"x_recon": x_recon, "z": z}
        return fwd_rtn

    def forward_discrim(self, x):
        [encoder.eval() for encoder in self.encoders]
        z = self.encode(x)
        d_real, d_fake = self.disc(z)
        fwd_rtn = {"d_real": d_real, "d_fake": d_fake, "z": z}
        return fwd_rtn

    def forward_gen(self, x):
        [encoder.train() for encoder in self.encoders]
        self.discriminator.eval()
        z = self.encode(x)
        _, d_fake = self.disc(z)
        fwd_rtn = {"d_fake": d_fake, "z": z}
        return fwd_rtn

    def recon_loss(self, x, fwd_rtn):
        x_recon = fwd_rtn["x_recon"]
        recon = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                recon += compute_mse(x[i], x_recon[i][j])
        return recon / self.n_views / self.n_views

    def generator_loss(self, fwd_rtn):
        z = fwd_rtn["z"]
        d_fake = fwd_rtn["d_fake"]
        gen_loss = 0
        label_real = np.zeros((z[0].shape[0], self.n_views + 1))
        label_real[:, 0] = 1
        label_real = torch.FloatTensor(label_real).to(self.device)
        for i in range(self.n_views):
            gen_loss += -torch.mean(label_real * torch.log(d_fake[i] + self.eps))
        return gen_loss / self.n_views

    def discriminator_loss(self, fwd_rtn):
        z = fwd_rtn["z"]
        d_real = fwd_rtn["d_real"]
        d_fake = fwd_rtn["d_fake"]
        disc_loss = 0
        label_real = np.zeros((z[0].shape[0], self.n_views + 1))
        label_real[:, 0] = 1
        label_real = torch.FloatTensor(label_real).to(self.device)

        disc_loss += -torch.mean(label_real * torch.log(d_real + self.eps))
        for i in range(self.n_views):
            label_fake = np.zeros((z[0].shape[0], self.n_views + 1))
            label_fake[:, i + 1] = 1
            label_fake = torch.FloatTensor(label_fake).to(self.device)
            disc_loss += -torch.mean(label_fake * torch.log(d_fake[i] + self.eps))

        return disc_loss / (self.n_views + 1)