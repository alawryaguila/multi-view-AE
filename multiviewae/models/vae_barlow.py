import torch
import hydra

from ..base.constants import MODEL_VAEBARLOW
from ..base.base_model import BaseModelVAE

class VAE_barlow(BaseModelVAE):
    """
    Multi-view Variational Autoencoder model with barlow twins loss between latent representations.

    Args:
    cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
        
        - model.beta (int, float): KL divergence weighting term.
        - model.alpha (int, float): Barlow twins weighting term.
        - model.sparse (bool): Whether to enforce sparsity of the encoding distribution.
        - model.threshold (float): Dropout threshold applied to the latent dimensions. Default is 0.
        - encoder.default._target_ (multiviewae.architectures.mlp.VariationalEncoder): Type of encoder class to use.
        - encoder.default.enc_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoding distribution.
        - decoder.default._target_ (multiviewae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
        - decoder.default.init_logvar (int, float): Initial value for log variance of decoder.
        - decoder.default.dec_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoding distribution.
    
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
        super().__init__(model_name=MODEL_VAEBARLOW,
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
            qz_xs (list): list of encoding dimensions for each view.
        """
        qz_xs = []
        for i in range(self.n_views):
            mu, logvar = self.encoders[i](x[i])
            qz_x = hydra.utils.instantiate(
                eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu, scale=logvar.exp().pow(0.5)
            )
            qz_xs.append(qz_x)
        return qz_xs

    def decode(self, qz_xs):
        r"""Forward pass of the latent dimensions through their respective decoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a single element, the inner list is a n_view element list with the position in the list indicating the decoder index.
        """  
        px_zs = []
        for i in range(self.n_views):
            px_z = self.decoders[i](qz_xs[i]._sample(training=self._training))
            px_zs.append(px_z)
        return [px_zs]

    def forward(self, x):
        r"""Apply encode and decode methods to input data to generate the latent dimensions and data reconstructions. 
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.
        """
        qz_xs = self.encode(x)
        px_zs = self.decode(qz_xs)
        fwd_rtn = {"px_zs": px_zs, "qz_xs": qz_xs}
        return fwd_rtn

    def calc_kl(self, qz_xs):
        r"""Calculate KL-divergence loss.

        Args:
            qz_xs (list): Single element list containing joint encoding distribution.

        Returns:
            (torch.Tensor): KL-divergence loss.
        """
        kl = 0
        for qz_x in qz_xs:
            kl += qz_x.kl_divergence(self.prior).mean(0).sum()
        return kl
    
    def calc_ll(self, x, px_zs):
        r"""Calculate log-likelihood loss.

        Args:
            x (list): list of input data of type torch.Tensor.
            px_zs (list): list of decoding distributions.

        Returns:
            ll (torch.Tensor): Log-likelihood loss.
        """
        ll = 0
        for i in range(self.n_views):
            ll += px_zs[0][i].log_likelihood(x[i]).mean(0).sum() #first index is latent, second index is view 
        return ll

    def calc_barlow_twins_loss(self, qz_xs):
        r"""Calculate barlow twins loss.

        Args:
            qz_xs (list): list of encoding distributions.

        Returns:
            (dict): Dictionary containing each element of the barlow twins loss.
        """
        smps = []
        for i, qz_x in enumerate(qz_xs):
            smp = qz_x._sample(training=self._training)
            smp_norm = self.bns[i](smp)
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
        r"""Calculate loss.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing each element of the loss.
        """
        px_zs = fwd_rtn["px_zs"]
        qz_xs = fwd_rtn["qz_xs"]

        kl = self.calc_kl(qz_xs)
        ll = self.calc_ll(x, px_zs)
        rtn = self.calc_barlow_twins_loss(qz_xs)
        btn = rtn["btn"]
        invariance = rtn["invariance"]
        covariance = rtn["covariance"]

        total = self.beta*kl - ll + self.alpha*btn

        losses = {"loss": total, "kl": kl, "ll": ll, "btn": btn, "invariance": invariance, "covariance": covariance}
        return losses
