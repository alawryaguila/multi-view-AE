import torch
from ..base.constants import MODEL_JOINTAAE, EPS
from ..base.base_model import BaseModelAAE

class jointAAE(BaseModelAAE):
    r"""
    Multi-view Adversarial Autoencoder model with a joint latent representation.

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            
            - discriminator._target_ (multiviewae.architectures.mlp.Discriminator): Discriminator network class.
            - discriminator.hidden_layer_dim (list): Number of nodes per hidden layer.
            - discriminator.bias (bool): Whether to include a bias term in hidden layers.
            - discriminator.non_linear (bool): Whether to include a ReLU() function between layers.
            - discriminator.dropout_threshold (float): Dropout threshold of layers.
        
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.
    """
    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        super().__init__(model_name=MODEL_JOINTAAE,
                cfg=cfg,
                input_dim=input_dim,
                z_dim=z_dim)

    def encode(self, x):
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            z (list): Single element list of joint latent dimensions of type torch.Tensor.
        """
        z = []
        for i in range(self.n_views):
            z_ = self.encoders[i](x[i])
            z.append(z_)

        z = torch.stack(z)
        mean_z = torch.mean(z, axis=0)
        return [mean_z]


    def decode(self, z):
        r"""Forward pass through decoder networks. The joint latent dimensions are passed through all of the decoders.

        Args:
            z (list): Single element list of joint latent dimensions of type torch.Tensor.

        Returns:
            px_zs (list): list of decoding distributions.
        """
        px_zs = []
        for i in range(self.n_views):
            px_z = self.decoders[i](z[0])
            px_zs.append(px_z)
        return [px_zs]

    def disc(self, z):
        r"""Forward pass of "real" samples from gaussian prior and "fake" samples from encoders through the discriminator network.

        Args:
            z (list): Single element list of joint latent dimensions of type torch.Tensor.

        Returns:
            d_real (torch.Tensor): Discriminator network output for "real" samples.
            d_fake (torch.Tensor): Discriminator network output for "fake" samples.
        """
        sh = z[0].shape
        z_real = self.prior.sample(sample_shape=sh)
        d_real = self.discriminator(z_real)
        d_fake = self.discriminator(z[0])
        return d_real, d_fake

    def forward_recon(self, x):
        r"""Apply encode and decode methods to input data to generate latent dimensions and data reconstructions.
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing decoding distributions (px_zs) and latent dimensions (z).
        """
        z = self.encode(x)
        px_zs = self.decode(z)
        fwd_rtn = {"px_zs": px_zs, "z": z}
        return fwd_rtn

    def forward_discrim(self, x):
        r"""Apply encode and disc methods to input data to generate discriminator prediction on the latent dimensions and train discriminator parameters.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing discriminator output from "real" samples (d_real), discriminator output from "fake" samples (d_fake), and latent dimensions (z).
        """
        [encoder.eval() for encoder in self.encoders]
        z = self.encode(x)
        d_real, d_fake = self.disc(z)
        fwd_rtn = {"d_real": d_real, "d_fake": d_fake, "z": z}
        return fwd_rtn

    def forward_gen(self, x):
        r"""Apply encode and disc methods to input data to generate discriminator prediction on the latent dimensions and train encoder parameters.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): fwd_rtn (dict): dictionary containing discriminator output from "fake" samples (d_fake) and latent dimensions (z).
        """
        [encoder.train() for encoder in self.encoders]
        self.discriminator.eval()
        z = self.encode(x)
        _, d_fake = self.disc(z)
        fwd_rtn = {"d_fake": d_fake, "z": z}
        return fwd_rtn

    def recon_loss(self, x, fwd_rtn):
        r"""Calculate reconstruction loss.

        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): fwd_rtn from the forward_recon method.

        Returns:
            ll (torch.Tensor): Reconstruction error.
        """
        px_zs = fwd_rtn["px_zs"]
        ll = 0
        for i in range(self.n_views):
            ll += - px_zs[0][i].log_likelihood(x[i]).mean(0).sum() #first index is latent, second index is view
        return ll / self.n_views

    def generator_loss(self, fwd_rtn):
        r"""Calculate the generator loss.

        Args:
            fwd_rtn (dict): fwd_rtn from the forward_gen method.

        Returns:
            gen_loss (torch.Tensor): Generator loss.
        """
        d_fake = fwd_rtn["d_fake"]
        gen_loss = torch.mean(1 - torch.log(d_fake + EPS))
        return gen_loss

    def discriminator_loss(self, fwd_rtn):
        r"""Calculate the discriminator loss.

        Args:
            fwd_rtn (dict): fwd_rtn from the forward_discrim method.

        Returns:
            disc_loss (torch.Tensor): Discriminator loss.
        """
        z = fwd_rtn["z"]
        d_real = fwd_rtn["d_real"]
        d_fake = fwd_rtn["d_fake"]
        disc_loss = -torch.mean(
            torch.log(d_real + EPS) + torch.log(1 - d_fake + EPS)
        )
        return disc_loss
