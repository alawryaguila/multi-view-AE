from ..base.constants import MODEL_AE
from ..base.base_model import BaseModelAE

class AE(BaseModelAE):
    r"""Multi-view Autoencoder model with a separate latent representation for each view.

    Args:
        cfg (str): Path to configuration file. 
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions. 
    """
    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_AE,
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
        x_recon = []
        for i in range(self.n_views):
            temp_recon = [self.decoders[j](z[i]) for j in range(self.n_views)]
            x_recon.append(temp_recon)
        return x_recon

    def forward(self, x):
        self.zero_grad()
        z = self.encode(x)
        x_recon = self.decode(z)
        fwd_rtn = {"x_recon": x_recon, "z": z}
        return fwd_rtn

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn["x_recon"]
        recon = self.recon_loss(x, x_recon)
        losses = {"loss": recon}
        return losses

    def recon_loss(self, x, x_recon):
        recon = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                recon += x_recon[j][i].log_likelihood(x[i]).sum(1, keepdims=True).mean(0) #first index is latent, second index is view
        return recon / self.n_views / self.n_views
