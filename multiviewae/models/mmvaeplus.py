import math
import torch
import hydra

from ..base.constants import MODEL_MMVAEPLUS
from ..base.base_model import BaseModelVAE
from ..base.distributions import Default
import numpy as np
#set numpy seed
np.random.seed(0)

#TODO: check if private priors are used at all here??

class mmVAEPlus(BaseModelVAE):
    r"""
    Mixture-of-Experts Multimodal Variational Autoencoder (MMVAE). 

    Code is based on: https://github.com/iffsid/mmvae

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            
            - model.K (int): Number of samples to take from encoding distribution.
            - model.DREG_loss (bool): Whether to use DReG estimator when using large K value. 
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
        super().__init__(model_name=MODEL_MMVAEPLUS,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)
    
        #create the prior parameters
        self.mean_priors_private = []
        self.logvar_priors_private = []

        for i in range(self.n_views):
            #set parameters of private priors
            mean_prior = torch.nn.Parameter(torch.zeros(1, self.w_dim), requires_grad=False).to(self.device)
            logvar_prior = torch.nn.Parameter(torch.zeros(1, self.w_dim), requires_grad=self.learn_private_prior).to(self.device)
            self.mean_priors_private.append(mean_prior)
            self.logvar_priors_private.append(logvar_prior)
        #set parameters of shared priors
        mean_prior = torch.nn.Parameter(torch.zeros(1, self.u_dim+self.w_dim), requires_grad=False).to(self.device) 
        logvar_prior = torch.nn.Parameter(torch.zeros(1, self.u_dim+self.w_dim), requires_grad=self.learn_shared_prior).to(self.device) 
        self.mean_priors_shared = mean_prior
        self.logvar_priors_shared = logvar_prior

    def encode(self, x):
        r"""Forward pass through encoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): list of encoding distributions for shared (qu_xs) and private (qw_xs) latent dimensions during training, otherwise return samples from encoding distributions.
        """
        qw_xs = []
        qu_xs = []
        for i in range(self.n_views):
            mu_u, logvar_u, mu_w, logvar_w = self.encoders[i](x[i])
            #private latent distribution
            qw_x = hydra.utils.instantiate(
                eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_w, logvar=logvar_w
            )
            #shared latent distribution
            qu_x = hydra.utils.instantiate(
                eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_u, logvar=logvar_u
            )

            qw_xs.append(qw_x)
            qu_xs.append(qu_x)
        if self._training:
            return [qw_xs, qu_xs]
        zss = []
        for i in range(self.n_views):

            u_x = qu_xs[i]._sample() #shared sample
            w_x = qw_xs[i]._sample() #private sample
            z_x = torch.cat((u_x, w_x), dim=-1)
            z_x = Default(x=z_x)

            zss.append(z_x)
        return zss
    
    def encode_subset(self, x, subset):
        r"""Forward pass through encoder networks for a subset of modalities. For modalities not in subset, shared latents are sampled from a random modality and 
        private latents from the shared prior.

        Args:
            x (list): list of input data of type torch.Tensor.
            subset (list): list of modalities to encode.

        Returns:
            (list): list of samples from encoding distributions.
        """    
        qw_xs = []
        qu_xs = []

        for i in range(self.n_views):
            if i in subset:
                mu_u, logvar_u, mu_w, logvar_w = self.encoders[i](x[i])
                #private latent distribution
                qw_x = hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_w, logvar=logvar_w
                )
                #shared latent distribution
                qu_x = hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_u, logvar=logvar_u
                )
            else: 
                # Choose one of the subset modalities at random
                mod = np.random.choice(subset)
                mu_u, logvar_u, mu_w, logvar_w = self.encoders[mod](x[mod])
                #shared latent distribution
                qu_x = hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{mod}.enc_dist"), loc=mu_u, logvar=logvar_u
                )
                #private latent distribution
                #sample private latent from shared prior
                qw_x = hydra.utils.instantiate(self.cfg.prior, loc=self.mean_priors_shared[:,self.u_dim:], logvar=self.logvar_priors_shared[:,self.u_dim:])
            qw_xs.append(qw_x)
            qu_xs.append(qu_x)
        if self._training:
            return [qw_xs, qu_xs]
        zss = []
        for i in range(self.n_views):
            w_x = qw_xs[i]._sample() #private sample
            u_x = qu_xs[i]._sample() #shared sample
            shape_ = list(w_x.shape)
            shape_[0] = u_x.shape[0]
            w_x = w_x.expand(shape_)
            z_x = torch.cat((u_x, w_x), dim=-1)
            z_x = Default(x=z_x)
            zss.append(z_x)
        return zss
    
    def decode(self, zss):
        r"""Forward pass through decoder networks. Each latent is passed through all of the decoders.

        Args:
            zss (list): list of latent samples if not training or list containing qw_xs (list of private encoding distributions) and
            qu_xs (list of shared encoding distributions) if training.

        Returns:
            (list): A nested list of decoding distributions. The outer list has a n_view element indicating latent dimensions index. 
            The inner list is a n_view element list with the position in the list indicating the decoder index.
        """    
        px_zs = []
        if self._training:
            qw_xs, qu_xs = zss[0], zss[1]
            for i in range(self.n_views):
                px_zs_inner = []
                for j in range(self.n_views): 
                    if i == j:
                        u_x = qu_xs[i].rsample(torch.Size([self.K])) #shared sample
                        w_x = qw_xs[j].rsample(torch.Size([self.K])) #private sample
                        z_x = torch.cat((u_x, w_x), dim=-1)
                    else:
                        u_x = qu_xs[i].rsample(torch.Size([self.K])) #shared sample
                        prior = hydra.utils.instantiate(
                            eval(f"self.cfg.encoder.enc{j}.enc_dist"), loc=self.mean_priors_private[j], logvar=self.logvar_priors_private[j]
                        )
                        w = prior.rsample(torch.Size([self.K])).to(self.device) #private sample
                        w = w.expand(w_x.shape)
                        z_x = torch.cat((u_x, w), dim=-1)
                    px_z = self.decoders[j](z_x)
                    px_zs_inner.append(px_z)
                px_zs.append(px_zs_inner)
        else:
            for i in range(self.n_views):
                px_zs_inner = []
                for j in range(self.n_views): 
                    z_x = zss[i]._sample(training=self._training, return_mean=self.return_mean)
                    if i != j:
                        u_x = z_x[:,:self.u_dim]
                        prior = hydra.utils.instantiate(self.cfg.prior, loc=self.mean_priors_shared[:,self.u_dim:], logvar=self.logvar_priors_shared[:,self.u_dim:])
                        w = prior.rsample().to(self.device) #private sample
                        shape_ = list(w.shape)
                        shape_[0] = u_x.shape[0]
                        w = w.expand(shape_)

                        z_x = torch.cat((u_x, w), dim=-1)
                    px_z = self.decoders[j](z_x) 
                    px_zs_inner.append(px_z)
                px_zs.append(px_zs_inner)            
        return px_zs

    def decode_subset(self, zss, subset):
        r"""Forward pass through decoder networks for a subset of modalities. Each latent is passed through its own decoder.

        Args:
            zss (list): list of latent samples for each modality.
            subset (list): list of modalities to decode.

        Returns:
            (list): A list of decoding distributions for each modality in subset.
        """    
        px_zs = []
        for i, zs in enumerate(zss):
            if i in subset:
                px_z = self.decoders[i](zs._sample(training=self._training, return_mean=self.return_mean))
                px_zs.append(px_z)
        return [px_zs]

    def forward(self, x):
        r"""Apply encode and decode methods to input data to generate latent dimensions and data reconstructions. 
        
        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing encoding (qw_xs and qu_xs) and decoding (px_zs) distributions.
        """
        if self._training:
            qw_xs, qu_xs = self.encode(x)[0], self.encode(x)[1]
            px_zs = self.decode([qw_xs, qu_xs])
        else:
            zss = self.encode(x)
            px_zs = self.decode(zss)
        return {"qw_xs": qw_xs, "qu_xs": qu_xs, "px_zs": px_zs}

    def loss_function(self, x, fwd_rtn):
        r"""Wrapper function for mmVAE loss.

        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            losses (dict): dictionary containing mmVAE loss.
        """
        qw_xs, qu_xs, px_zs = fwd_rtn["qw_xs"], fwd_rtn["qu_xs"], fwd_rtn["px_zs"]
        total = -self.moe_iwae(x, qw_xs, qu_xs, px_zs)
        losses = {"loss": total}
        return losses

    def moe_iwae(self, x, qw_xs, qu_xs, px_zs):
        r"""Calculate Mixture-of-Experts importance weighted autoencoder (IWAE) loss used for the mmVAE model.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing encoding and decoding distributions.

        Returns:
            (torch.Tensor): the output tensor.
        """
        lws = []
        prior = hydra.utils.instantiate(self.cfg.prior, loc=self.mean_priors_shared, logvar=self.logvar_priors_shared)
 
        for i in range(self.n_views):
            if self._training:
                us = qu_xs[i].rsample(torch.Size([self.K]))
                ws = qw_xs[i].rsample(torch.Size([self.K]))
                zs = torch.cat((us, ws), dim=-1)
            else:
                us = qu_xs[i]._sample()
                ws = qw_xs[i]._sample()
                zs = torch.cat((us, ws), dim=-1)

            lpz = prior.log_likelihood(zs).sum(-1)
            lqu_x = self.log_mean_exp(
                torch.stack([qu_x.log_likelihood(us).sum(-1) for qu_x in qu_xs])
            )  # summing over M modalities for each z to create q(u|x1:M)
            
            lqw_x = qw_xs[i].log_likelihood(ws).sum(-1)

            lpx_z = [
                px_z.log_likelihood(x[d]).view(*px_z._sample().size()[:2], -1).sum(-1)
                for d, px_z in enumerate(px_zs[i])
            ]  # summing over each decoder
            lpx_z = torch.stack(lpx_z).sum(0)

            lw = lpx_z + self.beta*(lpz - lqu_x - lqw_x)
            lws.append(lw)

        return (
            self.log_mean_exp(torch.stack(lws), dim=1).mean(0).sum()/self.n_views
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
