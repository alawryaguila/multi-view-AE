import torch
import hydra

from ..base.constants import MODEL_MMJSD, EPS
from ..base.base_model import BaseModelVAE
from ..base.representations import MixtureOfExperts, alphaProductOfExperts

class mmJSD(BaseModelVAE):
    r"""
    Multimodal Jensen-Shannon divergence (mmJSD) model with Product-of-Experts dynamic prior. 

    Code is based on: https://github.com/thomassutter/mmjsd

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            
            - model.private (bool): Whether to include private modality-specific latent dimensions.
            - model.s_dim (int): Number of private latent dimensions.
            - encoder.default._target_ (multiviewae.architectures.mlp.VariationalEncoder): Type of encoder class to use.
            - encoder.default.enc_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoding distribution.
            - decoder.default._target_ (multiviewae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
            - decoder.default.init_logvar(int, float): Initial value for log variance of decoder.
            - decoder.default.dec_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoding distribution.
    
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.

    References
    ----------
    Sutter, Thomas & Daunhawer, Imant & Vogt, Julia. (2021). Multimodal Generative Learning Utilizing Jensen-Shannon-Divergence. Advances in Neural Information Processing Systems. 33. 

    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_MMJSD,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)


    def encode(self, x):
        r"""Forward pass through encoder network. If self.private=True, the first two dimensions of each latent are used for the modality-specific part and the remaining dimensions for the joint content.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            Returns a combination of the following depending on the training stage and model type: 
            qz_xs (list): Single elemet list containing the MoE encoding distribution for self.private=False.
            qzs_xs (list): list containing each encoding distribution for self.private=False.
            qz_x (list):  Single element list containing the PoE encoding distribution for self.private=False.

            qc_x (list): Single element list containing the PoE shared encoding distribution.
            qscs_xs (list): list containing combined shared and private latents.
            qs_xs (list): list of encoding distributions for private latents.
            qcs_xs (list): list containing encoding distributions for shared latent dimensions for each view.
        """
     
        if self.private:
            qs_xs = []
            qcs_xs = []
            mu_s = []
            logvar_s = []
            mu_c = []
            logvar_c = []
            for i in range(self.n_views):
                mu, logvar = self.encoders[i](x[i])
                mu_s.append(mu[:,:self.s_dim])
                logvar_s.append(logvar[:,:self.s_dim])
                mu_c.append(mu[:,self.s_dim:])
                logvar_c.append(logvar[:,self.s_dim:])

                qs_x = hydra.utils.instantiate(
                eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu[:,:self.s_dim], scale=logvar[:,:self.s_dim].exp().pow(0.5)+EPS
                )
                qs_xs.append(qs_x)
                qc_x = hydra.utils.instantiate(
                eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu[:,self.s_dim:], scale=logvar[:,self.s_dim:].exp().pow(0.5)+EPS
                )
                qcs_xs.append(qc_x)

            mu_c = torch.stack(mu_c)
            logvar_c = torch.stack(logvar_c)

            moe_mu_c, moe_logvar_c = MixtureOfExperts()(mu_c, logvar_c)
       
            poe_mu_c, poe_logvar_c = alphaProductOfExperts()(mu_c, logvar_c)
            qc_x = hydra.utils.instantiate( 
            self.cfg.encoder.default.enc_dist, loc=poe_mu_c, scale=poe_logvar_c.exp().pow(0.5)+EPS
            )
            qscs_xs = []
            for i in range(self.n_views):       
                mu_sc = torch.cat((mu_s[i], moe_mu_c), 1)
                logvar_sc = torch.cat((logvar_s[i], moe_logvar_c), 1)
                qsc_x = hydra.utils.instantiate( 
                eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_sc, scale=logvar_sc.exp().pow(0.5)+EPS
                )
                qscs_xs.append(qsc_x)

            if self._training:
                return [[qc_x], qcs_xs, qs_xs, qscs_xs]
            return qscs_xs   

        mu = []
        logvar = []
        qzs_xs = []
        for i in range(self.n_views):
            mu_, logvar_ = self.encoders[i](x[i])
            mu.append(mu_) 
            logvar.append(logvar_) 
            qz_x = hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_, scale=logvar_.exp().pow(0.5)+EPS
            )
            qzs_xs.append(qz_x)

        mu = torch.stack(mu)
        logvar = torch.stack(logvar)

        moe_mu, moe_logvar = MixtureOfExperts()(mu, logvar)

        qz_xs =  hydra.utils.instantiate(
                self.cfg.encoder.default.enc_dist, loc=moe_mu, scale=moe_logvar.exp().pow(0.5)+EPS
            )
        if self._training:
            poe_mu, poe_logvar = alphaProductOfExperts()(mu, logvar)
            qz_x = hydra.utils.instantiate( 
                self.cfg.encoder.default.enc_dist, loc=poe_mu, scale=poe_logvar.exp().pow(0.5)+EPS
            )
            return [[qz_xs], qzs_xs, [qz_x]]

        return [qz_xs]

    def decode(self, qz_x):
        r"""Forward pass of latent dimensions through decoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a single element, the inner list is a n_view element list with the position in the list indicating the decoder index.
        """      
        px_zs = []
        for i in range(self.n_views):
            if self.private:
                px_z = self.decoders[i](qz_x[i]._sample(training=self._training))
            else:
                px_z = self.decoders[i](qz_x[0]._sample(training=self._training))
            px_zs.append(px_z)
        return [px_zs]

    def forward(self, x):
        r"""Apply encode and decode methods to input data to generate the joint and modality specific latent dimensions and data reconstructions. 
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.
        """
        if self.private:
            qc_x, qcs_xs, qs_xs, qscs_xs = self.encode(x)
            px_zs = self.decode(qscs_xs)
            fwd_rtn = {"px_zs": px_zs, "qcs_xs": qcs_xs, "qs_xs": qs_xs, "qc_x": qc_x}
        else:
            qz_xs, qzs_xs, qz_x = self.encode(x)
            px_zs = self.decode(qz_xs)
            fwd_rtn = {"px_zs": px_zs, "qzs_xs": qzs_xs, "qz_x": qz_x}
        return fwd_rtn
        
    def calc_kl(self, qz_xs):
        r"""Calculate KL-divergence loss.

        Args:
            qz_xs (list): list of encoding distributions.

        Returns:
            (torch.Tensor): KL-divergence loss.
        """
        kl = 0
        for i in range(len(qz_xs)):
            kl += qz_xs[i].kl_divergence(self.prior).sum(1, keepdims=True).mean(0)
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
            ll += px_zs[0][i].log_likelihood(x[i]).mean(0) .sum()#first index is latent, second index is view 
        return ll

    def calc_jsd(self, qcs_xs, qc_x):
        r"""Calculate Jensen-Shannon Divergence loss.

        Args:
            qcs_xs (list): list of encoding distributions of each view for shared latent dimensions.
            qc_x (list): Dynamic prior given by PoE of shared encoding distributions.

        Returns:
            jsd (torch.Tensor): Jensen-Shannon Divergence loss.
        """
        jsd = 0
        for i in range(self.n_views):
            jsd += qcs_xs[i].kl_divergence(qc_x[0]).sum(1, keepdims=True).mean(0)
        return jsd

    def loss_function(self, x, fwd_rtn):
        r"""Calculate mmJSD loss.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing each element of the mmJSD loss.
        """
        px_zs = fwd_rtn["px_zs"]
        ll = self.calc_ll(x, px_zs)

        if self.private:
            qcs_xs = fwd_rtn["qcs_xs"]
            qs_xs = fwd_rtn["qs_xs"] 
            qc_x = fwd_rtn["qc_x"]
            jsd = self.calc_jsd(qcs_xs, qc_x)
            kl = self.calc_kl(qs_xs)
            total = kl + jsd - ll
            losses = {"loss": total, "kl": kl, "ll": ll, "jsd": jsd}
        else:
            qz_x = fwd_rtn["qz_x"]
            qzs_xs = fwd_rtn["qzs_xs"]

            jsd = self.calc_jsd(qzs_xs, qz_x)
            total = jsd - ll
            losses = {"loss": total, "ll": ll, "jsd": jsd}
        return losses
