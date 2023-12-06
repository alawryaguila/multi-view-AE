import torch
import hydra
import numpy as np
from ..base.constants import MODEL_JMVAE
from ..base.base_model import BaseModelVAE

class JMVAE(BaseModelVAE):
    r"""
    JMVAE-kl.

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            - alpha (float): Weighting of KL-divergence loss from individual encoders.
            - encoder.default._target_ (multiviewae.architectures.mlp.VariationalEncoder): Type of Encoder to use.
            - encoder.default.enc_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoding distribution.
            - decoder.default._target_ (multiviewae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
            - decoder.default.init_logvar(int, float): Initial value for log variance of decoder.
            - decoder.default.dec_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoding distribution.
        
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.

    References
    ----------
    Suzuki, Masahiro & Nakayama, Kotaro & Matsuo, Yutaka. (2016). Joint Multimodal Learning with Deep Generative Models.
    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_JMVAE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

    def _setencoders(self):
        r"""Set the joint and individual encoder networks.
        """
        self.encoders = torch.nn.ModuleList(
              [hydra.utils.instantiate( 
                    self.cfg.encoder.default,
                    input_dim=self.input_dim[0]+self.input_dim[1],
                    z_dim=self.z_dim,
                    sparse=False,
                    log_alpha=None,
                    _recursive_=False,
                    _convert_="all"
                )]
                +
            [
                hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{i}"),
                    input_dim=d,
                    z_dim=self.z_dim,
                    sparse=False,
                    log_alpha=None,
                    _recursive_=False,
                    _convert_="all"
                )
                for i, d in enumerate(self.input_dim)
            ]
        )

    def encode(self, x):
        r"""Forward pass through joint encoder network. 

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): Single element list containing joint encoding distribution, qz_xy.
        """
        mu, logvar = self.encoders[0](torch.cat((x[0], x[1]),dim=1)) #TODO: current implementation only works for 2D data for both modalities
        qz_xy = hydra.utils.instantiate(  
            self.cfg.encoder.default.enc_dist, loc=mu, logvar=logvar
        )
        return [qz_xy]

    def encode_separate(self, x):
        r"""Forward pass through separate encoder networks. 

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            qz_x: Encoding distribution for modality X.
            qz_y: Encoding distribution for modality Y.
        """
        mu, logvar = self.encoders[1](x[0])
        qz_x = hydra.utils.instantiate(    
            self.cfg.encoder.enc0.enc_dist, loc=mu, logvar=logvar
        )
        mu, logvar = self.encoders[2](x[1])
        qz_y = hydra.utils.instantiate(   
            self.cfg.encoder.enc1.enc_dist, loc=mu, logvar=logvar
        )
        return qz_x, qz_y

    def decode(self, qz_x):
        r"""Forward pass of joint latent dimensions through decoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a single element indicating the shared latent dimensions. 
            The inner list is a 2 element list with the position in the list indicating the decoder index.
        """       
        px_zs = []
        for i in range(self.n_views):
            px_z = self.decoders[i](qz_x[0]._sample(training=self._training, return_mean=self.return_mean))
            px_zs.append(px_z)
        return [px_zs]

    def forward(self, x):
        r"""Apply encode and decode methods to input data to generate latent dimensions and data reconstructions. 
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.
        """
        qz_xy = self.encode(x)
        qz_x, qz_y = self.encode_separate(x)
        px_z, py_z = self.decode(qz_xy)[0]
        fwd_rtn = {"px_z": px_z, "py_z": py_z, "qz_x": qz_x, "qz_y": qz_y, "qz_xy": qz_xy}
        return fwd_rtn

    def calc_kl(self, qz_xy, qz_x, qz_y):
        r"""Calculate JMVAE-kl KL-divergence loss.

        Args:
            qz_xy (list): Single element list containing shared encoding distribution.
            qz_x (list): Single element list containing encoding distribution for modality X.
            qz_y (list): Single element list containing encoding distribution for modality Y.


        Returns:
            (torch.Tensor): KL-divergence loss.
        """

        kl_prior = qz_xy[0].kl_divergence(self.prior).mean(0).sum()
        kl_qz_x = qz_xy[0].kl_divergence(qz_x).mean(0).sum()
        kl_qz_y = qz_xy[0].kl_divergence(qz_y).mean(0).sum()

        return kl_prior + self.alpha*(kl_qz_x + kl_qz_y)

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
            ll += px_zs[i].log_likelihood(x[i]).mean(0).sum()
        return ll

    def loss_function(self, x, fwd_rtn):
        r"""Calculate JMVAE-kl loss.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing each element of the JMVAE loss.
        """
        px_z = fwd_rtn["px_z"]
        py_z = fwd_rtn["py_z"]
        qz_x = fwd_rtn["qz_x"]
        qz_y = fwd_rtn["qz_y"]
        qz_xy = fwd_rtn["qz_xy"]

        kls = self.calc_kl(qz_xy, qz_x, qz_y)
        ll = self.calc_ll(x, [px_z, py_z])

        if self.current_epoch > self.warmup - 1:
            annealing_factor = 1
        else:
            annealing_factor = self.current_epoch/self.warmup

        total = annealing_factor*kls - ll

        losses = {"loss": total, "kls": kls, "ll": ll}
        return losses

    def calc_nll(self, x, K=1000, batch_size_K=100):
        r"""Calculate negative log-likelihood used to evaluate model performance.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            nll (torch.Tensor): Negative log-likelihood.
        """
        mu, logvar = self.encoders[0](torch.cat((x[0], x[1]),dim=1)) #TODO: current implementation only works for 2D data for both modalities
        qz_xy = hydra.utils.instantiate(  
            self.cfg.encoder.default.enc_dist, loc=mu, logvar=logvar
        )
        # And sample from the posterior
        zs = qz_xy.rsample([K])  # shape K x n_data x latent_dim
        ll = 0
        start_idx = 0
        stop_idx = min(start_idx + batch_size_K, K)
        lnpxs = []
        while start_idx < stop_idx:
            zs_ = zs[start_idx:stop_idx]
            lpx_zs = 0
            for j in range(self.n_views):
                px_z = self.decoders[j](zs_)
                lpx_zs += px_z.log_likelihood(x[j]).sum(dim=-1)

            lpz = self.prior.log_likelihood(zs_).sum(dim=-1)
            lqz_xy = qz_xy.log_likelihood(zs_).sum(dim=-1)
            ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=0)
            lnpxs.append(ln_px)
            start_idx += batch_size_K
            stop_idx = min(stop_idx + batch_size_K, K)
        lnpxs = torch.stack(lnpxs)
        ll += (torch.logsumexp(lnpxs, dim=0) - np.log(K)).mean()
        return -ll