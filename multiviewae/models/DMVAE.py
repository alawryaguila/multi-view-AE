import torch
import hydra

from ..base.constants import MODEL_DMVAE, EPS
from ..base.base_model import BaseModelVAE
from ..base.representations import ProductOfExperts

class DMVAE(BaseModelVAE):
    r"""
    Disentangled multi-modal variational autoencoder (DMVAE)

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            
            - model._lambda (list, optional): Log likelihood weighting term for each modality.
            - model.s_dim (int): Number of private latent dimensions.
            - encoder.default._target_ (multiviewae.architectures.mlp.VariationalEncoder): Type of encoder class to use.
            - encoder.default.enc_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoding distribution.
            - decoder.default._target_ (multiviewae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
            - decoder.default.init_logvar (int, float): Initial value for log variance of decoder.
            - decoder.default.dec_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoding distribution.
        
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.

    References
    ----------
    M. Lee and V. Pavlovic, "Private-Shared Disentangled Multimodal VAE for Learning of Latent Representations," 2021 IEEE/CVF Conference on Computer Vision and 
    Pattern Recognition Workshops (CVPRW), Nashville, TN, USA, 2021, pp. 1692-1700, doi: 10.1109/CVPRW53098.2021.00185.

    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_DMVAE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

        self.join_z = ProductOfExperts()
        #if lambda is none, set to 1 for all modalities
        if not hasattr(self, '_lambda'):
            self._lambda = [1 for i in range(self.n_views)]
        else:
            assert len(self._lambda) == self.n_views, "Number of lambda weightings must be equal to number of views"
        print('_lambda: ', self._lambda)

    def encode(self, x):
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            qc_x (list): Single element list containing the PoE shared encoding distribution.
            qcs_xs (list): list containing encoding distributions for shared latent dimensions for each view.
            qs_xs (list): list of encoding distributions for private latents.
            qscs_xs (list): nested list containing combined PoE shared and private latents.
            qscss_xs (list): nested list containing combined shared latents from each modality and private latents for same and cross view reconstruction.
            
        """
        qs_xs = []
        qcs_xs = []
        qscs_xs = []
        qscss_xs = []
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

        for i in range(self.n_views):
            qscs_x_ = []
            for j in range(self.n_views):
                mu_sc = torch.cat((mu_s[i], mu_c[j]), 1) #when i /= j for recon of view from shared latent of other view
                logvar_sc = torch.cat((logvar_s[i], logvar_c[j]), 1)
                qscs_x = hydra.utils.instantiate(
                eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_sc, scale=logvar_sc.exp().pow(0.5)+EPS
                )
                qscs_x_.append(qscs_x)
            qscss_xs.append(qscs_x_)

        mu_c = torch.stack(mu_c)
        logvar_c = torch.stack(logvar_c)
    
        mu_c, logvar_c = self.join_z(mu_c, logvar_c)
        qc_x = hydra.utils.instantiate(
            self.cfg.encoder.default.enc_dist, loc=mu_c, scale=logvar_c.exp().pow(0.5)+EPS
        )

        for i in range(self.n_views):   
            mu_sc = torch.cat((mu_s[i], mu_c), 1)
            logvar_sc = torch.cat((logvar_s[i], logvar_c), 1)
            qsc_x = hydra.utils.instantiate( 
            eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_sc, scale=logvar_sc.exp().pow(0.5)+EPS
            )
            qscs_xs.append(qsc_x)

        if self._training:
            return [[qc_x], qcs_xs, qs_xs, qscs_xs, qscss_xs]
        return qscs_xs


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
            px_z = self.decoders[i](qz_x[i]._sample(training=self._training))
            px_zs.append(px_z)
        return [px_zs]

    def decode_separate(self, qz_xs):
        r"""Forward pass through decoder networks. Each shared latent is passed through all of the decoders with the private latents from the same view.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a n_view element list with position in the list indicating the decoder index. 
            The inner list is a n_view element list with the position in the list indicating latent dimensions index. NOTE: This is the reverse to other models.
        """  
        px_zs = []
        for i in range(self.n_views):
            px_z = [
                self.decoders[i](qz_xs[j][i]._sample(training=self._training)) #TODO: check this is right
                for j in range(self.n_views)
            ]
            px_zs.append(px_z)
        return px_zs 

    def forward(self, x):
        r"""Apply encode and decode methods to input data to generate the latent dimensions and data reconstructions. 
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.
        """
        qc_x, qcs_xs, qs_xs, qscs_xs, qscss_xs = self.encode(x)
        px_zs = self.decode(qscs_xs)
        pxs_zs = self.decode_separate(qscss_xs)
        fwd_rtn = {"px_zs": px_zs, "pxs_zs": pxs_zs, "qcs_xs": qcs_xs, "qs_xs": qs_xs, "qc_x": qc_x}

        return fwd_rtn


    def calc_kl_joint_latent(self, qz_x, qs_xs):
        r"""Calculate KL-divergence loss for the first terms in Equation 3.

        Args:
            qz_x (list): Single element list containing joint encoding distribution.
            qs_xs (list): list of encoding distributions for private latent dimensions for each view.

        Returns:
            (torch.Tensor): KL-divergence loss.
        """
        kl = 0
        for i in range(self.n_views):
            kl += qs_xs[i].kl_divergence(self.prior).mean(0).sum()
            kl += qz_x[0].kl_divergence(self.prior).mean(0).sum()

        return kl
    
    def calc_kl_separate_latent(self, qcs_xs, qs_xs):
        r"""Calculate KL-divergence loss for the second terms in Equation 3.

        Args:
            qcs_x (list): list of the shared encoding distributions calculated from each view.
            qs_xs (list): list of encoding distributions for private latent dimensions for each view.

        Returns:
            (torch.Tensor): KL-divergence loss.
        """
        kl = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                kl += qcs_xs[j].kl_divergence(self.prior).mean(0).sum() 
                kl += qs_xs[i].kl_divergence(self.prior).mean(0).sum()
        return kl 
    
    def calc_ll_joint(self, x, px_zs):
        r"""Calculate log-likelihood loss from the joint encoding distribution for each modality. 

        Args:
            x (list): list of input data of type torch.Tensor.
            px_zs (list): list of decoding distributions. 

        Returns:
            ll (torch.Tensor): Log-likelihood loss.
        """
        ll = 0
        for i in range(self.n_views):
            ll += self._lambda[i]*px_zs[0][i].log_likelihood(x[i]).mean(0).sum() #first index is latent, second index is view
        return ll

    def calc_ll_separate(self, x, pxs_zs):
        r"""Calculate cross-modal and self-reconstrution log-likelihood loss from the shared encoding distribution for each modality and private latents.

        Args:
            x (list): list of input data of type torch.Tensor.
            pxs_zs (list): nested list of decoding distributons. NOTE: The ordering of decoding distribution is the reverse compared to other models.

        Returns:
            ll (torch.Tensor): Log-likelihood loss.
        """
        ll = 0
        for i in range(self.n_views):
            for j in range(self.n_views):
                ll += self._lambda[i]*pxs_zs[i][j].log_likelihood(x[i]).mean(0).sum() #first index is view, second index is latent
        return ll    
               
    def loss_function(self, x, fwd_rtn):
        r"""Calculate DMVAE loss.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing each element of the DMVAE loss.
        """
        px_zs = fwd_rtn["px_zs"]
        ll = self.calc_ll_joint(x, px_zs)

        qc_x = fwd_rtn["qc_x"]
        qcs_xs = fwd_rtn["qcs_xs"]
        qs_xs = fwd_rtn["qs_xs"]
        pxs_zs = fwd_rtn["pxs_zs"] 

        kl = self.calc_kl_joint_latent(qc_x, qs_xs) 
        kl += self.calc_kl_separate_latent(qcs_xs, qs_xs) 
        ll += self.calc_ll_separate(x, pxs_zs) 
        
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

        return optimizers