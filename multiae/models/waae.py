import torch

from torch.autograd import Variable

from ..base.constants import MODEL_WAAE
from ..base.base_model import BaseModelAAE

class wAAE(BaseModelAAE):
    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        super().__init__(model_name=MODEL_WAAE,
                cfg=cfg,
                input_dim=input_dim,
                z_dim=z_dim)

        self.is_wasserstein = True

    def encode(self, x):
        z = []
        for i in range(self.n_views):
            z_ = self.encoders[i](x[i])
            z.append(z_)

        z = torch.stack(z)
        mean_z = torch.mean(z, axis=0)
        return [mean_z]

    def decode(self, z):
        x_out = []
        for i in range(self.n_views):
            x_ = self.decoders[i](z[0])
            x_out.append([x_])
        return x_out

    def disc(self, z):
        z_real = Variable(torch.randn(z[0].size()[0], self.z_dim) * 1.0).to(self.device)
        d_real = self.discriminator(z_real)
        d_fake = self.discriminator(z[0])
        return d_real, d_fake

    def forward_recon(self, x):
        z = self.encode(x)
        x_out = self.decode(z)
        fwd_rtn = {"x_recon": x_out, "z": z}
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
            recon += x_recon[i][0].log_likelihood(x[i]).sum(1, keepdims=True).mean(0)
        return recon / self.n_views

    def generator_loss(self, fwd_rtn):
        z = fwd_rtn["z"]
        d_fake = fwd_rtn["d_fake"]
        gen_loss = -torch.mean(d_fake.sum(dim=-1))
        return gen_loss

    def discriminator_loss(self, fwd_rtn):
        z = fwd_rtn["z"]
        d_real = fwd_rtn["d_real"]
        d_fake = fwd_rtn["d_fake"]

        disc_loss = -torch.mean(d_real.sum(dim=-1)) + torch.mean(d_fake.sum(dim=-1))

        return disc_loss
