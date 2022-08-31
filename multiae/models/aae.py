import torch
import numpy as np
from torch.autograd import Variable
from ..base.distributions import Normal
from ..base.constants import MODEL_AAE
from ..base.base_model import BaseModelAAE

class AAE(BaseModelAAE):
    """
    Multi-view Adversarial Autoencoder model with a separate latent representation for each view.

    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        super().__init__(model_name=MODEL_AAE,
                cfg=cfg,
                input_dim=input_dim,
                z_dim=z_dim)

    def encode(self, x):
        z = []
        for i in range(self.n_views):
            z_ = self.encoders[i](x[i])
            z.append(z_)
        return z

    def decode(self, z):
        px_zs = []
        for i in range(self.n_views):
            px_z = [self.decoders[i](z[j]) for j in range(self.n_views)]
            px_zs.append(px_z)
        return px_zs

    def disc(self, z):
        sh = z[0].shape
        z_real = self.prior.sample(sample_shape=sh)
        d_real = self.discriminator(z_real)
        d_fake = []
        for i in range(self.n_views):
            d = self.discriminator(z[i])
            d_fake.append(d)
        return d_real, d_fake

    def forward_recon(self, x):
        z = self.encode(x)
        px_zs = self.decode(z)
        fwd_rtn = {"px_zs": px_zs, "z": z}
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
        px_zs = fwd_rtn["px_zs"]
        ll = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                ll += - px_zs[i][j].log_likelihood(x[i]).sum(1, keepdims=True).mean(0)
        return ll / self.n_views / self.n_views

    def generator_loss (self, fwd_rtn):
        d_fake = fwd_rtn["d_fake"]
        gen_loss = 0
        for i in range(self.n_views):
            gen_loss += torch.mean(1 - torch.log(d_fake[i] + self.eps))
        return gen_loss/self.n_views
    
    def discriminator_loss(self, fwd_rtn):
        d_real = fwd_rtn["d_real"]
        d_fake = fwd_rtn["d_fake"]

        disc_loss = -torch.mean(torch.log(d_real + self.eps)) 
        for i in range(self.n_views):
            disc_loss += -torch.mean(1 - torch.log(d_fake[i] + self.eps))
        return disc_loss / (self.n_views + 1)


