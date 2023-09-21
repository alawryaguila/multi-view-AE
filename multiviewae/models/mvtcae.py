import torch
import hydra

from ..base.constants import MODEL_MVTCAE, EPS
from ..base.base_model import BaseModelVAE
from ..base.representations import ProductOfExperts

class mvtCAE(BaseModelVAE):
    r"""
    Multi-View Total Correlation Auto-Encoder (MVTCAE).

    Code is based on: https://github.com/gr8joo/MVTCAE

    NOTE: This implementation currently only caters for a PoE posterior distribution. MoE and MoPoE posteriors will be included in further work.

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            
            - model.beta (int, float): KL divergence weighting term.
            - model.alpha (int, float): Log likelihood, Conditional VIB and VIB weighting term.
            - encoder.default._target_ (multiviewae.architectures.mlp.VariationalEncoder): Type of encoder class to use.
            - encoder.default.enc_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoding distribution.
            - decoder.default._target_ (multiviewae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
            - decoder.default.init_logvar(int, float): Initial value for log variance of decoder.
            - decoder.default.dec_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoding distribution.
    
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.

    References
    ----------
    Hwang, HyeongJoo and Kim, Geon-Hyeong and Hong, Seunghoon and Kim, Kee-Eung. Multi-View Representation Learning via Total Correlation Objective. 2021. NeurIPS

    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        super().__init__(model_name=MODEL_MVTCAE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

    def encode(self, x):
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            Returns either the joint or separate encoding distributions depending on whether the model is in the training stage: 
            qz_xs (list): list containing separate encoding distributions.
            qz_x (list): Single element list containing PoE joint encoding distribution.
        """
     
        if self._training:
            qz_xs = []
            for i in range(self.n_views):
                mu, logvar = self.encoders[i](x[i])
                qz_x = hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu, scale=logvar.exp().pow(0.5)+EPS
                )
                qz_xs.append(qz_x)
            return qz_xs
        else:
            mu = []
            logvar = []
            for i in range(self.n_views):
                mu_, logvar_ = self.encoders[i](x[i])
                mu.append(mu_)
                logvar.append(logvar_)
            mu = torch.stack(mu)

            logvar = torch.stack(logvar)
            mu, logvar = ProductOfExperts()(mu, logvar)
            qz_x = hydra.utils.instantiate( 
                self.cfg.encoder.default.enc_dist, loc=mu, scale=logvar.exp().pow(0.5)
            )
            qz_x = [qz_x]
            return qz_x

    def decode(self, qz_xs):
        r"""Forward pass of joint latent dimensions through decoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a single element indicating the shared latent dimensions. 
            The inner list is a n_view element list with the position in the list indicating the decoder index.
        """  
        if self._training:
            mu = [qz_x.loc for qz_x in qz_xs]
            var = [qz_x.variance for qz_x in qz_xs]
            mu = torch.stack(mu)
            var = torch.stack(var)
            mu, logvar = ProductOfExperts()(mu, torch.log(var))
            px_zs = []
            for i in range(self.n_views):
                px_z = self.decoders[i](
                    hydra.utils.instantiate(
                    self.cfg.encoder.default.enc_dist, loc=mu, scale=logvar.exp().pow(0.5)+EPS
                    ).rsample()
                )
                px_zs.append(px_z)
            return [px_zs]
        else:
            px_zs = []
            for i in range(self.n_views):
                px_z = self.decoders[i](qz_xs[0].loc)
                px_zs.append(px_z)
            return [px_zs]

    def forward(self, x):
        r"""Apply encode and decode methods to input data to generate the joint latent dimensions and data reconstructions. 
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.
        """
        qz_xs = self.encode(x)
        px_zs = self.decode(qz_xs)
        fwd_rtn = {"px_zs": px_zs, "qz_xs": qz_xs}
        return fwd_rtn

    def loss_function(self, x, fwd_rtn):
        r"""Calculate MVTCAE loss.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing each element of the MVTCAE loss.
        """
        px_zs = fwd_rtn["px_zs"]
        qz_xs = fwd_rtn["qz_xs"]

        rec_weight = (self.n_views - self.alpha) / self.n_views
        cvib_weight = self.alpha / self.n_views
        vib_weight = 1 - self.alpha

        grp_kl = self.calc_kl_groupwise(qz_xs)
        cvib_kl = self.calc_kl_cvib(qz_xs)
        ll = self.calc_ll(x, px_zs)

        kld_weighted = cvib_weight * cvib_kl + vib_weight * grp_kl
        total = -rec_weight * ll + self.beta * kld_weighted

        losses = {"loss": total, "kl_cvib": cvib_kl, "kl_grp": grp_kl, "ll": ll}
        return losses

    def calc_kl_cvib(self, qz_xs):
        r"""Calculate KL-divergence between PoE joint encoding distribution and the encoding distribution for each view.

        Args:
            qz_xs (list): list of encoding distributions of each view.

        Returns:
            kl (torch.Tensor): KL-divergence loss.
        """
        mu = [qz_x.loc for qz_x in qz_xs]
        var = [qz_x.variance for qz_x in qz_xs]
        mu = torch.stack(mu)
        var = torch.stack(var)
        mu, logvar = ProductOfExperts()(mu, torch.log(var))
        kl = 0
        for i in range(self.n_views):
            kl += (
                hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu, scale=logvar.exp().pow(0.5)
                )
                .kl_divergence(qz_xs[i]).sum(1, keepdims=True).mean(0)
            )
        return kl

    def calc_kl_groupwise(self, qz_xs):
        r"""Calculate KL-divergence between the encoding distribution for each view and the prior distribution.

        Args:
            qz_xs (list): list of encoding distributions of each view.

        Returns:
            kl (torch.Tensor): KL-divergence loss.
        """
        mu = [qz_x.loc for qz_x in qz_xs]
        var = [qz_x.variance for qz_x in qz_xs]
        mu = torch.stack(mu)
        var = torch.stack(var)
        mu, logvar = ProductOfExperts()(mu, torch.log(var))
        return (
            hydra.utils.instantiate(
                self.cfg.encoder.default.enc_dist, loc=mu, scale=logvar.exp().pow(0.5)
            )
            .kl_divergence(self.prior).sum(1, keepdims=True).mean(0)
        )

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
