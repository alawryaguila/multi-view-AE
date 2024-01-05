from ..base.constants import MODEL_DCCAE, EPS
from ..base.base_model import BaseModelAE
import torch

class DCCAE(BaseModelAE):
    r"""Deep Canonically Correlated Autoencoder (DCCAE).
    CCA implementation adapted from: https://github.com/jameschapman19/cca_zoo

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            
            - model._lambda (int, float): Reconstruction weighting term

        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.

    References
    ----------
    Wang, Weiran & Arora, Raman & Livescu, Karen & Bilmes, Jeff. (2016). On Deep Multi-View Representation Learning: Objectives and Optimization.

    """
    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_DCCAE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

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
        r"""Forward pass through decoder networks. Each latent is passed through all of the decoders.

        Args:
            z (list): list of latent dimensions for each view of type torch.Tensor.

        Returns:
            x_recon (list): list of data reconstructions.
        """
        x_recon = []
        for i in range(self.n_views):
            temp_recon = self.decoders[i](z[i])
            x_recon.append(temp_recon)
        return [x_recon]

    def cca(self, zs):
        r"""CCA loss calculation.
        Adapted from: https://github.com/jameschapman19/cca_zoo/blob/main/cca_zoo/deep/_discriminative/_dmcca.py

        Args:
            z (list): list of latent dimensions for each view of type torch.Tensor.

        Returns:
            cca_loss (torch.Tensor): CCA loss.
        """

        zs = [
            z - z.mean(dim=0)
            for z in zs
        ]
        all_views = torch.cat(zs, dim=1)

        #Calculate cross-covariance matrix
        C = torch.cov(all_views.T)
        C = C - torch.block_diag(
            *[torch.cov(z.T) for z in zs]
        )
        C =  C / len(zs)

        #Calculate block covariance matrix
        D = torch.block_diag(
            *[
                (1 - EPS) * torch.cov(z.T)
                + EPS
                * torch.eye(z.shape[1], device=z.device)
                for z in zs
            ]
        )
        D = D / len(zs)

        C += D

        U, S, V = torch.svd(D)
        # Enforce positive definite by taking a torch max() with EPS
        S = torch.clamp(S, min=EPS)
        # Calculate inverse square-root
        inv_sqrt_S = torch.diag_embed(torch.pow(S, -0.5))
        # Calculate inverse square-root matrix
        R = torch.matmul(torch.matmul(U, inv_sqrt_S), V.transpose(-1, -2))

        C_whitened = R @ C @ R.T
        eigvals = torch.linalg.eigvalsh(C_whitened)
        idx = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[idx[:self.z_dim]]
        eigvals = torch.nn.LeakyReLU()(eigvals[torch.gt(eigvals, 0)])
        corr = eigvals.sum()

        return -corr
    
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
        recon = 0
        for i in range(self.n_views):
            recon += - x_recon[0][i].log_likelihood(x[i]).mean(0).sum() #first index is latent, second index is view

        cca_loss = self.cca(z)        
        total_loss = self._lambda * recon + cca_loss
        losses = {"loss": total_loss, "recon_loss": recon, "cca_loss": cca_loss}
        return losses