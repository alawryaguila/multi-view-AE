import math
import torch
import hydra

from ..base.constants import MODEL_MMVAE, EPS
from ..base.base_model import BaseModelVAE

class mmVAE(BaseModelVAE):
    r"""
    Mixture-of-Experts Multimodal Variational Autoencoder (MMVAE). 

    Code is based on: https://github.com/iffsid/mmvae

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            
            - model.K (int): Number of samples to take from encoding distribution.
            - encoder.default._target_ (multiviewae.architectures.mlp.VariationalEncoder): Type of encoder class to use.
            - encoder.default.enc_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoding distribution.
            - decoder.default._target_ (multiviewae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
            - decoder.default.init_logvar (int, float): Initial value for log variance of decoder.
            - decoder.default.dec_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoding distribution.
            
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions. 
    
    References
    ----------
    Shi, Y., Siddharth, N., Paige, B., & Torr, P.H. (2019). Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models. ArXiv, abs/1911.03393.
    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_MMVAE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

    def encode(self, x):
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): list of encoding distributions.
        """
        qz_xs = []
        for i in range(self.n_views):
            mu, logvar = self.encoders[i](x[i])
            qz_x = hydra.utils.instantiate(
                eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu, scale=logvar.exp().pow(0.5)+EPS
            )
            qz_xs.append(qz_x)
        return qz_xs

    def decode(self, qz_xs):
        r"""Forward pass through decoder networks. Each latent is passed through all of the decoders.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions. The outer list has a n_view element indicating latent dimensions index. 
            The inner list is a n_view element list with the position in the list indicating the decoder index.
        """    
        px_zs = []
        for qz_x in qz_xs:
            if self._training:
                px_z = [
                    self.decoders[j](qz_x.rsample(torch.Size([self.K])))
                    for j in range(self.n_views)
                ]
            else:
                px_z = [
                    self.decoders[j](qz_x.rsample())
                    for j in range(self.n_views)
                ]
            px_zs.append(
                px_z
            )  
            del px_z
        return px_zs

    def forward(self, x):
        r"""Apply encode and decode methods to input data to generate latent dimensions and data reconstructions. 
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding (qz_xs) and decoding (px_zs) distributions.
        """
        qz_xs = self.encode(x)
        px_zs = self.decode(qz_xs)
        return {"qz_xs": qz_xs, "px_zs": px_zs}

    def loss_function(self, x, fwd_rtn):
        r"""Wrapper function for mmVAE loss.

        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing mmVAE loss.
        """
        qz_xs, px_zs = fwd_rtn["qz_xs"], fwd_rtn["px_zs"]
        total = -self.moe_iwae(x, qz_xs, px_zs)
        losses = {"loss": total}
        return losses

    def moe_iwae(self, x, qz_xs, px_zs):
        r"""Calculate Mixture-of-Experts importance weighted autoencoder (IWAE) loss used for the mmVAE model.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            (torch.Tensor): the output tensor.
        """
        lws = []
        zss = []
        for i in range(self.n_views):
            if self._training:
                zs = qz_xs[i].rsample(torch.Size([self.K]))
            else:
                zs = qz_xs[i].rsample()
            zss.append(zs)

        for r, qz_x in enumerate(qz_xs): 
            lpz = self.prior.log_likelihood(zss[r]).sum(-1)
            lqz_x = self.log_mean_exp(
                torch.stack([qz_x.log_likelihood(zss[r]).sum(-1) for qz_x in qz_xs])
            )  # summing over M modalities for each z to create q(z|x1:M)
            
            lpx_z = [
                px_z.log_likelihood(x[d]).view(*px_z._sample().size()[:2], -1).sum(-1)
                for d, px_z in enumerate(px_zs[r])
            ]  # summing over each decoder
            lpx_z = torch.stack(lpx_z).sum(0)

            lw = lpz + lpx_z - lqz_x
            lws.append(lw)
        return (
            self.log_mean_exp(torch.stack(lws), dim=1).mean(0).sum()
        )  # looser iwae bound 

    def log_mean_exp(self, value, dim=0, keepdim=False):
        r"""Returns the log of the mean of the exponentials along the given dimension (dim).

        Args:
            value (torch.Tensor): the input tensor.
            dim (int, optional): the dimension along which to take the mean.
            keepdim (bool, optional): whether the output tensor has dim retained or not.

        Returns:
            (torch.Tensor): the output tensor.
        """
        return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))
