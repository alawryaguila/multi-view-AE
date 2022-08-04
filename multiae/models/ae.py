import torch
from ..base.base_model import BaseModel
from ..utils.calc_utils import compute_mse, update_dict
import hydra

class AE(BaseModel):
    def __init__(
        self,
        input_dims,
        model="AE",
        network=None,
        **kwargs,
    ):

        super().__init__(model=model, network=network)

        self.save_hyperparameters()

        self.__dict__.update(self.cfg.model)
        self.__dict__.update(kwargs)

        self.cfg.encoder = update_dict(self.cfg.encoder, kwargs)
        self.cfg.decoder = update_dict(self.cfg.decoder, kwargs)

        self.model_type = model
        self.input_dims = input_dims
        self.n_views = len(input_dims)

        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    self.cfg.encoder,
                    _recursive_=False,
                    input_dim=input_dim,
                    z_dim=self.z_dim,
                )
                for input_dim in self.input_dims
            ]
        )
        self.decoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    self.cfg.decoder,
                    _recursive_=False,
                    input_dim=input_dim,
                    z_dim=self.z_dim,
                )
                for input_dim in self.input_dims
            ]
        )

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                list(self.encoders[i].parameters())
                + list(self.decoders[i].parameters()),
                lr=self.learning_rate,
            )
            for i in range(self.n_views)
        ]
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

    def sample_from_dist(self, dist):
        return dist._sample()
        
    def forward(self, x):
        self.zero_grad()
        z = self.encode(x)
        x_recon = self.decode(z)
        fwd_rtn = {"x_recon": x_recon, "z": z}
        return fwd_rtn

    def recon_loss(self, x, x_recon):
        recon = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                recon += x_recon[i][j].log_likelihood(x[i]).sum(1, keepdims=True).mean(0)
        return recon / self.n_views / self.n_views

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn["x_recon"]
        recon = self.recon_loss(x, x_recon)
        losses = {"loss": recon}
        return losses
