import torch
import hydra

from ..base.constants import MODEL_WEIGHTEDMVAE, EPS
from ..base.base_model import BaseModelVAE
from ..base.representations import weightedProductOfExperts

class weighted_mVAE(BaseModelVAE):
    r"""
    Generalised Product-of-Experts Variational Autoencoder (gPoE-MVAE).

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            
            - model.beta (int, float): KL divergence weighting term.
            - model.private (bool): Whether to include private view-specific latent dimensions.
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
    Cao, Y., & Fleet, D. (2014). Generalized Product of Experts for Automatic and Principled Fusion of Gaussian Process Predictions. arXiv.
    Lawry Aguila, A., Chapman, J., Altmann, A. (2023). Multi-modal Variational Autoencoders for normative modelling across multiple imaging modalities. arXiv

    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_WEIGHTEDMVAE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

        self.join_z = weightedProductOfExperts()
        #check if self.private is attribute, if not set to False
        if not hasattr(self, 'private'):
            self.private = False
        if self.private:
            tmp_weight = torch.FloatTensor(len(input_dim), self.z_dim - self.s_dim).fill_(1/len(input_dim))
            self.poe_weight = torch.nn.Parameter(data=tmp_weight, requires_grad=True)
        else:
            tmp_weight = torch.FloatTensor(len(input_dim), self.z_dim).fill_(1/len(input_dim))
            self.poe_weight = torch.nn.Parameter(data=tmp_weight, requires_grad=True)

    
    def encode(self, x):
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            Returns a combination of the following depending on the training stage and model type: 
            qz_x (list):  Single element list containing the PoE encoding distribution for self.private=False.

            qc_x (list): Single element list containing the PoE shared encoding distribution.
            qs_xs (list): list of encoding distributions for private latents.
            qscs_xs (list): list containing combined shared and private latents.
            qcs_xs (list): list containing encoding distributions for shared latent dimensions for each view.
        """
        if not hasattr(self, 'private'):
            self.private = False
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
       
            mu_c, logvar_c = self.join_z(mu_c, logvar_c, self.poe_weight)
            qc_x = hydra.utils.instantiate(
                self.cfg.encoder.default.enc_dist, loc=mu_c, scale=logvar_c.exp().pow(0.5)+EPS
            )
            with torch.no_grad():
                self.poe_weight = self.poe_weight.clamp_(0, +1)
            qscs_xs = []
            for i in range(self.n_views):       
                mu_sc = torch.cat((mu_s[i], mu_c), 1)
                logvar_sc = torch.cat((logvar_s[i], logvar_c), 1)
                qsc_x = hydra.utils.instantiate( 
                eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_sc, scale=logvar_sc.exp().pow(0.5)+EPS
                )
                qscs_xs.append(qsc_x)

            if self._training:
                return [[qc_x], qcs_xs, qs_xs, qscs_xs]
            return qscs_xs

        mu = []
        logvar = []
        for i in range(self.n_views):
            mu_, logvar_ = self.encoders[i](x[i])
            mu.append(mu_)
            logvar.append(logvar_)
        mu = torch.stack(mu)
        logvar = torch.stack(logvar)
        mu_out, logvar_out = self.join_z(mu, logvar, self.poe_weight)
        qz_x = hydra.utils.instantiate(
            self.cfg.encoder.default.enc_dist, loc=mu_out, scale=logvar_out.exp().pow(0.5)+EPS
        )
        with torch.no_grad():
            self.poe_weight = self.poe_weight.clamp_(0, +1)
        return [qz_x]

    def decode(self, qz_x):
        r"""Forward pass of joint latent dimensions through decoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a single element indicating the shared latent dimensions. 
            The inner list is a n_view element list with the position in the list indicating the decoder index.
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
        r"""Apply encode and decode methods to input data to generate the joint latent dimensions and data reconstructions. 
        
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
            qz_x = self.encode(x)
            px_zs = self.decode(qz_x)
            fwd_rtn = {"px_zs": px_zs, "qz_x": qz_x}
        return fwd_rtn

    def calc_kl(self, qz_x):
        r"""Calculate KL-divergence loss.

        Args:
            qz_x (list): Single element list containing joint encoding distribution.

        Returns:
            (torch.Tensor): KL-divergence loss.
        """
        kl = qz_x[0].kl_divergence(self.prior).mean(0).sum()
        return self.beta * kl
    
    def calc_kl_separate(self, qc_xs):
        r"""Calculate KL-divergence loss.

        Args:
            qc_xs (list): list of encoding distributions for private/shared latent dimensions for each view.

        Returns:
            (torch.Tensor): KL-divergence loss.
        """
        kl = 0
        for i in range(self.n_views):
            kl += qc_xs[i].kl_divergence(self.prior).mean(0).sum()
        return self.beta * kl
    
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

    def loss_function(self, x, fwd_rtn):
        r"""Calculate Multimodal VAE loss.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing each element of the MVAE loss.
        """
        px_zs = fwd_rtn["px_zs"]
        ll = self.calc_ll(x, px_zs)
        if self.private:
            qs_xs = fwd_rtn["qs_xs"]
            qc_x = fwd_rtn["qc_x"]
            kl = self.calc_kl_separate(qs_xs) #calc kl for private latents
            kl += self.calc_kl(qc_x) #calc kl for shared latents
            total = kl - ll
            losses = {"loss": total, "kl": kl, "ll": ll}
            return losses
        else:
            qz_x = fwd_rtn["qz_x"]
            kl = self.calc_kl(qz_x)
            total = kl - ll
            losses = {"loss": total, "kl": kl, "ll": ll}
            return losses

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                list(self.encoders[i].parameters())
                + list(self.decoders[i].parameters()), 
                lr=self.learning_rate,
            )
            for i in range(self.n_views)
        ]
        optimizers.append(torch.optim.Adam([self.poe_weight],
                lr=self.learning_rate
            ))
        return optimizers
