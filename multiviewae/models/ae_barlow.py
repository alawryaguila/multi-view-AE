from ..base.constants import MODEL_AEBARLOW
from ..base.base_model import BaseModelAE
import torch
class AE_barlow(BaseModelAE):
    """
    Multi-view Autoencoder model with barlow twins loss between latent representations.

    Args:
    cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:    

        - model.beta (int, float): KL divergence weighting term.
        - model.alpha (int, float): Barlow twins weighting term.

    input_dim (list): Dimensionality of the input data.
    z_dim (int): Number of latent dimensions.

    References
    ----------
    Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. International Conference on Machine Learning.
    Chapman et al., (2021). CCA-Zoo: A collection of Regularized, Deep Learning based, Kernel, and Probabilistic CCA methods in a scikit-learn style framework. Journal of Open Source Software, 6(68), 3823, https://doi.org/10.21105/joss.03823
    """
    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_AEBARLOW,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(self.z_dim, affine=False) for _ in self.encoders]
        )
    def encode(self, x):
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            z (list): list of latent dimensions for each view of type torch.Tensor.
        """   
        z = []
        for i in range(self.n_views):
            z_ = self.encoders[i](x[i])
            z.append(z_)
        return z

    def decode(self, z):
        r"""Forward pass through decoder networks. Each latent is passed through its respective decoder.

        Args:
            z (list): list of latent dimensions for each view of type torch.Tensor.

        Returns:
            x_recon (list): list of data reconstructions.
        """
        x_recon = []
        for i in range(self.n_views):
            temp_recon = self.decoders[i](z[i])
            x_recon.append(temp_recon)
        return x_recon
    
    def forward(self, x):
        r"""Apply encode and decode methods to input data to generate latent dimensions and data reconstructions.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing list of data reconstructions (x_recon) and latent dimensions (z).
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        fwd_rtn = {"x_recon": x_recon, "z": z}
        return fwd_rtn

    def calc_recon_loss(self, x, x_recon):
        r"""Calculate reconstruction loss.

        Args:
            x (list): list of input data of type torch.Tensor.
            x_recon (list): list of data reconstructions.

        Returns:
            (torch.Tensor): reconstruction loss.
        """
        recon = 0
        for i in range(self.n_views):
            recon += x_recon[i].log_likelihood(x[i]).mean(0).sum()
        return recon

    def calc_barlow_twins_loss(self, z):
        r"""Calculate barlow twins loss.

        Args:
            z (list): list of latent dimensions for each view of type torch.Tensor.
        Returns:
            (dict): Dictionary containing each element of the barlow twins loss.
        """
        smps = []
        for i, z_ in enumerate(z):
            smp_norm = self.bns[i](z_)
            smps.append(smp_norm)

        cross_cov = smps[0].T @ smps[1] / smps[0].shape[0]
        invariance = torch.sum(torch.pow(1 - torch.diag(cross_cov), 2))
        covariance = torch.sum(
            torch.triu(torch.pow(cross_cov, 2), diagonal=1)
        ) + torch.sum(torch.tril(torch.pow(cross_cov, 2), diagonal=-1))

        return {
            "btn": invariance + covariance,
            "invariance": invariance,
            "covariance": covariance,
        }
    
    def loss_function(self, x, fwd_rtn):    
        r"""Calculate reconstruction loss.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing list of data reconstructions (x_recon) and latent dimensions (z).

        Returns:
            losses (dict): dictionary containing reconstruction loss.
        """
        x_recon = fwd_rtn["x_recon"]
        z = fwd_rtn["z"]

        recon = self.calc_recon_loss(x, x_recon)
        btn = self.calc_barlow_twins_loss(z)

        loss = recon + self.alpha * btn["btn"]

        losses = {"loss": loss, "recon": recon, "btn": btn["btn"]}
        return losses
