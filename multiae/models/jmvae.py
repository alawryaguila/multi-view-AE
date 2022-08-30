import torch
import hydra

from ..base.constants import MODEL_JMVAE
from ..base.base_model import BaseModelVAE
from ..base.distributions import Normal
from ..base.exceptions import ModelInputError

class JMVAE(BaseModelVAE):
    """
    Implementation of JMVAE-kl from Joint Multimodal Learning with Deep Generative Models (https://arxiv.org/abs/1611.01891)
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
        assert len(input_dim)==2, 'JMVAE expects two input dimensions'

    def _setencoders(self):
        self.encoders = torch.nn.ModuleList(
              [hydra.utils.instantiate(
                    self.cfg.encoder,
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
                    self.cfg.encoder,
                    input_dim=d,
                    z_dim=self.z_dim,
                    sparse=False,
                    log_alpha=None,
                    _recursive_=False,
                    _convert_="all"
                )
                for d in self.input_dim
            ]
        )

    def _setdecoders(self):
        self.decoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    self.cfg.decoder,
                    input_dim=d,
                    z_dim=self.z_dim,
                    _recursive_=False,
                    _convert_ = "all"
                )
                for d in self.input_dim
            ]
        )


    def encode(self, x):
        mu, logvar = self.encoders[0](torch.cat((x[0], x[1]),dim=1))
        qz_xy = hydra.utils.instantiate(
            self.cfg.encoder.enc_dist, loc=mu, scale=logvar.exp().pow(0.5)
        )
        return [qz_xy]

    def encode_separate(self, x):
        mu, logvar = self.encoders[1](x[0])
        qz_x = hydra.utils.instantiate(
            self.cfg.encoder.enc_dist, loc=mu, scale=logvar.exp().pow(0.5)
        )
        mu, logvar = self.encoders[2](x[1])
        qz_y = hydra.utils.instantiate(
            self.cfg.encoder.enc_dist, loc=mu, scale=logvar.exp().pow(0.5)
        )
        return qz_x, qz_y

    def decode(self, qz_x):
        px_zs = []
        for i in range(self.n_views):
            px_z = self.decoders[i](qz_x[0]._sample(training=self._training))
            px_zs.append([px_z])
        return px_zs

    def decode_separate(self, qz_xs):
        px_zs = []
        for i in range(self.n_views):
            px_z = self.decoders[i](qz_xs[i]._sample(training=self._training))
            px_zs.append([px_z])
        return px_zs

    def forward(self, x):
        qz_xy = self.encode(x)
        qz_x, qz_y = self.encode_separate(x)
        px_z, py_z = self.decode_separate([qz_x, qz_y])
        fwd_rtn = {"px_z": px_z, "py_z": py_z, "qz_x": qz_x, "qz_y": qz_y, "qz_xy": qz_xy}
        return fwd_rtn
        
    def calc_kl(self, qz_xy, qz_x, qz_y):
        """
        VAE: Implementation from: https://arxiv.org/abs/1312.6114
        sparse-VAE: Implementation from: https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb
        """
        kl_prior = qz_xy[0].kl_divergence(self.prior).sum(1, keepdims=True).mean(0)
        kl_qz_x = qz_xy[0].kl_divergence(qz_x).sum(1, keepdims=True).mean(0)
        kl_qz_y = qz_xy[0].kl_divergence(qz_y).sum(1, keepdims=True).mean(0)

        return kl_prior + kl_qz_x + kl_qz_y

    def calc_ll(self, x, px_zs):
        ll = 0
        for i in range(self.n_views):
            ll += px_zs[i][0].log_likelihood(x[i]).sum(1, keepdims=True).mean(0)
        return ll

    def loss_function(self, x, fwd_rtn):

        px_z = fwd_rtn["px_z"]
        py_z = fwd_rtn["py_z"]
        qz_x = fwd_rtn["qz_x"]
        qz_y = fwd_rtn["qz_y"]
        qz_xy = fwd_rtn["qz_xy"]

        kls = self.calc_kl(qz_xy, qz_x, qz_y)
        ll = self.calc_ll(x, [px_z, py_z])

        total = kls - ll 

        losses = {"loss": total, "kls": kls, "ll": ll}
        return losses