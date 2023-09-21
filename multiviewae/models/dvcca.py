import torch
import hydra
from ..base.constants import MODEL_DVCCA, EPS
from ..base.base_model import BaseModelVAE

class DVCCA(BaseModelVAE):
    r"""Deep Variational Canonical Correlation Analysis (DVCCA).

    Args:
        cfg (str): Path to configuration file. Model specific parameters in addition to default parameters:
            
            - model.beta (int, float): KL divergence weighting term.
            - model.private (bool): Whether to include private view-specific latent dimensions.
            - model.sparse (bool): Whether to enforce sparsity of the encoding distribution.
            - model.threshold (float): Dropout threshold applied to the latent dimensions. Default is 0.
            - encoder.default._target_ (multiviewae.architectures.mlp.VariationalEncoder): Type of encoder class to use.
            - encoder.default.enc_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Encoding distribution.
            - decoder.default._target_ (multiviewae.architectures.mlp.VariationalDecoder): Type of decoder class to use.
            - decoder.default.init_logvar(int, float): Initial value for log variance of decoder.
            - decoder.default.dec_dist._target_ (multiviewae.base.distributions.Normal, multiviewae.base.distributions.MultivariateNormal): Decoding distribution.
        
        input_dim (list): Dimensionality of the input data.
        z_dim (int): Number of latent dimensions.

    References
    ----------
    Wang, Weiran & Lee, Honglak & Livescu, Karen. (2016). Deep Variational Canonical Correlation Analysis.


    """
    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        super().__init__(model_name=MODEL_DVCCA,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

    ################################            protected methods
    def _setencoders(self):
        r"""Set the encoder network using the first data input. If private=True also set a private encoder network for each view.
        """
        if self.sparse and self.threshold != 0.:
            self.log_alpha = torch.nn.Parameter(
                torch.FloatTensor(1, self.z_dim).normal_(0, 0.01)
            )
        else:
            self.sparse = False
            self.log_alpha = None

        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    self.cfg.encoder.default,  
                    input_dim=self.input_dim[0],
                    z_dim=self.z_dim,
                    sparse=self.sparse,
                    log_alpha=self.log_alpha,
                    _recursive_=False,
                    _convert_="all"
                )
            ]
        )

        if self.private:

            self.private_encoders = torch.nn.ModuleList(
                [
                    hydra.utils.instantiate(
                        eval(f"self.cfg.encoder.enc{i}"),
                        input_dim=d,
                        z_dim=self.z_dim,
                        sparse=self.sparse,
                        log_alpha=self.log_alpha,
                        _recursive_=False,
                        _convert_="all"
                    )
                    for i, d in enumerate(self.input_dim)
                ]
            )
            self.z_dim = self.z_dim + self.z_dim
            if self.sparse and self.threshold != 0.:

                self.log_alpha = torch.nn.Parameter(
                    torch.FloatTensor(1, self.z_dim).normal_(0, 0.01)
                )

    def configure_optimizers(self):
        r"""Configure optimizers for encoder, private encoder, and decoder network parameters.

        Returns:
            optimizers (list): list of Adam optimizers for encoders and decoders.
        """
        if self.private:
            optimizers = [
                torch.optim.Adam(self.encoders[0].parameters(), lr=self.learning_rate)
            ] + [
                torch.optim.Adam(
                    list(self.private_encoders[i].parameters()), lr=self.learning_rate
                )
                for i in range(self.n_views)
            ] + [
                torch.optim.Adam(
                    list(self.decoders[i].parameters()), lr=self.learning_rate
                )
                for i in range(self.n_views)
            ]
        else:
            optimizers = [
                torch.optim.Adam(self.encoders[0].parameters(), lr=self.learning_rate)
            ] + [
                torch.optim.Adam(
                    list(self.decoders[i].parameters()), lr=self.learning_rate
                )
                for i in range(self.n_views)
            ]
        return optimizers

    def encode(self, x):
        r"""Forward pass through encoder network. For DVCCA-private a forward pass is performed through each private encoder and the output latent is concatenated with the shared latent.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            Returns a combination of the following depending on the training stage and model type: 
            qz_x (list): list containing the shared encoding distribution.
            qz_xs (list): list of encoding distributions for shared and private latents of DVCCA-private.
            qh_xs (list): list of encoding distributions for private latents of DVCCA-private.

        """
        mu, logvar = self.encoders[0](x[0])
        qz_x = hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc0.enc_dist"), loc=mu, scale=logvar.exp().pow(0.5)+EPS
        )
        if self.private:
            qz_xs = []
            qh_xs = []
            for i in range(self.n_views):
                mu_p, logvar_p = self.private_encoders[i](x[i])
                qh_x = hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_p, scale=logvar_p.exp().pow(0.5)+EPS
                )
                qh_xs.append(qh_x)

                mu_ = torch.cat((mu, mu_p), 1)
                logvar_ = torch.cat((logvar, logvar_p), 1)

                qz_x = hydra.utils.instantiate(
                    eval(f"self.cfg.encoder.enc{i}.enc_dist"), loc=mu_, scale=logvar_.exp().pow(0.5)+EPS
                )
                qz_xs.append(qz_x)
            if self._training:
                return [[qz_x], qz_xs, qh_xs]
            return qz_xs
        else:
            qz_x = hydra.utils.instantiate( 
                self.cfg.encoder.default.enc_dist, loc=mu, scale=logvar.exp().pow(0.5)+EPS
            )
            return [qz_x]

    def decode(self, qz_x):
        r"""Forward pass through decoder networks.

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            (list): A nested list of decoding distributions, px_zs. The outer list has a single element indicating the shared or shared and private latent dimensions. 
            The inner list is a n_view element list with the position in the list indicating the decoder index.
        """
        px_zs = []
        for i in range(self.n_views):
            if self.private:
                x_out = self.decoders[i](qz_x[i]._sample(training=self._training))
            else:
                x_out = self.decoders[i](qz_x[0]._sample(training=self._training))
            px_zs.append(x_out)
        return [px_zs] 

    def forward(self, x):
        r"""Apply encode and decode methods to input data to generate latent dimensions and data reconstructions. 
        For DVCCA, the shared encoding distribution is passed to the decode method. 
        For DVCCA-private, the joint distribution of the shared and private latents for each view is passed to the decode method. 

        Args:
            x (list): list of input data of type torch.Tensor.

        Returns:
            fwd_rtn (dict): dictionary containing list of decoding distributions (px_zs), shared encoding distribution (qz_x), and (for DVCCA-private) private encoding distributions (qh_xs).
        """
        self.zero_grad()
        if self.private:
            qz_x, qz_xs, qh_xs = self.encode(x)
            px_zs = self.decode(qz_xs)
        else:
            qz_x = self.encode(x)
            px_zs = self.decode(qz_x)
            qh_xs = []
        fwd_rtn = {"px_zs": px_zs, "qz_x": qz_x, 'qh_xs': qh_xs}
        return fwd_rtn

    def loss_function(self, x, fwd_rtn):
        r"""Calculate DVCCA loss.
        
        Args:
            x (list): list of input data of type torch.Tensor.
            fwd_rtn (dict): dictionary containing list of decoding distributions (px_zs), shared encoding distribution (qz_x), and (for DVCCA-private) private encoding distributions (qh_xs).

        Returns:
            losses (dict): dictionary containing each element of the DVCCA loss.
        """
        px_zs = fwd_rtn["px_zs"]
        qz_x = fwd_rtn["qz_x"]
        qh_xs= fwd_rtn["qh_xs"]
        kl = self.calc_kl(qz_x, qh_xs)
        ll = self.calc_ll(x, px_zs)
        total = kl - ll
        losses = {"loss": total, "kl": kl, "ll": ll}
        return losses

    def calc_kl(self, qz_x, qh_xs):
        r"""Wrapper function for calculating KL-divergence loss.

        Args:
            qz_x (list): Single element list containing shared encoding distribution.
            qh_xs (list): list of  private encoding distributions for DVCCA-private.

        Returns:
            (torch.Tensor): KL-divergence loss across all views.
        """
        kl = 0
        kl += self.calc_kl_(qz_x[0])
        if self.private:      
            for i in range(self.n_views):
                kl+= self.calc_kl_(qh_xs[i])
        return self.beta * kl

    def calc_kl_(self, dist):
        r"""Calculate KL-divergence.

        Args:
            dist: Distribution object.

        Returns:
            (torch.Tensor): Kl-divergence.
        """
        if self.sparse:
            return dist.sparse_kl_divergence().mean(0).sum()
        return dist.kl_divergence(self.prior).mean(0).sum()

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
            ll += px_zs[0][i].log_likelihood(x[i]).mean(0).sum()  #first index is latent, second index is view
        return ll
