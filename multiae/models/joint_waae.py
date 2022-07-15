import torch
from .layers import Encoder, Decoder, Discriminator
from ..base.base_model import BaseModelAAE
from torch.autograd import Variable
from ..utils.calc_utils import compute_mse, update_dict
import hydra 

class wAAE(BaseModelAAE):
    def __init__(
        self,
        input_dims,
        expt='wAAE',
        **kwargs,
    ):
        super().__init__(expt=expt)

        self.save_hyperparameters()
        self.automatic_optimization = False

        self.__dict__.update(self.cfg.model)
        self.__dict__.update(kwargs)
        
        self.cfg.encoder = update_dict(self.cfg.encoder, kwargs)
        self.cfg.decoder = update_dict(self.cfg.decoder, kwargs)

        self.model_type = expt
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

        self.discriminator = Discriminator(
            input_dim=self.z_dim,
            hidden_layer_dims=self.discriminator_layer_dims,
            wasserstein=True,
            output_dim=1,
        )

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

        z = torch.stack(z)
        mean_z = torch.mean(z, axis=0)
        return z

    def decode(self, z):
        x_out = []
        for i in range(self.n_views):
            for j in range(self.n_views):
                x_ = self.decoders[i](z)
                x_out.append(x_)
        return x_out

    def disc(self, z):
        z_real = Variable(torch.randn(z[0].size()[0], self.z_dim) * 1.0).to(self.device)
        d_real = self.discriminator(z_real)
        d_fake = self.discriminator(z)
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
            recon += compute_mse(x[i], x_recon[i])
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