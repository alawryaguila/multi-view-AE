
import torch
from torch import nn
import linalg

class PCCA(nn.Module):
    def __init__(self, latent_dim, dims, max_iters, debug=False):

        self.latent_dim = latent_dim
        self.max_iters = max_iters
        self.debug = debug
        p1, p2 = dims 
        self.p1 = p1
        self.p2 = p2      
        Lambda1, Lambda2, B1, B2, log_Psi1_diag, log_Psi2_diag \
            = self.init_params()

        self.Lambda1 = nn.Parameter(Lambda1)
        self.Lambda2 = nn.Parameter(Lambda2)
        self.log_Psi1_diag = nn.Parameter(log_Psi1_diag)
        self.log_Psi2_diag = nn.Parameter(log_Psi2_diag)
        self.B1 = nn.Parameter(B1)
        self.B2 = nn.Parameter(B2)

    def forward(self, y):
        """Fit the probabilistic CCA model using Expectation-Maximization.
        :param y: Observations of shape (n_features, n_samples).
        """
        k = 3 * self.latent_dim #why?

        Lambda, Psi_diag = self.em_bishop(y)
        PLL_inv = linalg.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)
        z = self.E_z_given_y(Lambda, PLL_inv, y)

        yr = self.reparameterize(Lambda, Psi_diag, z)
        y1 = yr[:self.p1]
        y2 = yr[self.p1:]

        return y1.t(), y2.t()

    def em_bishop(self, y):
        """Perform Expectation-Maximization.
        :param y: Observations of shape (n_features, n_samples).
        :return:  Parameters that maximize the likelihood given the data.
        """
        Lambda, Psi_diag = self.tile_params()
        nlls = []

        for _ in range(self.max_iters):
            Lambda, Psi_diag = self.em_bishop_step(y, Lambda, Psi_diag)
            if self.debug:
                nll = self.neg_log_likelihood(y, Lambda, Psi_diag)
                nlls.append(nll)

        self.nlls      = nlls
        self.iters_req = self.max_iters
        return Lambda, Psi_diag

    def em_bishop_step(self, y, Lambda, Psi_diag):
        """Equations 5 and 6 in Ghahramani and Hinton (1996).
        :param y:        Observations of shape (n_features, n_samples).
        :param Lambda:   Current Lambda.
        :param Psi_diag: Current Psi_diag.
        :return:         Maximum likelihood parameters for the current step.
        """
        k = 3 * self.latent_dim
        p, n = y.shape

        PLL_inv = linalg.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)

        # E-step: compute expected moments for latent variable z.
        # -------------------------------------------------------
        Ez = self.E_z_given_y(Lambda, PLL_inv, y)
        Ezz = self.E_zzT_given_y(Lambda, PLL_inv, y, k)

        # M-step: compute optimal Lambda and Psi.
        # ---------------------------------------

        # Compute Lambda_new (Equation 5, G&H 1996).
        Lambda_lterm = linalg.sum_outers(y, Ez)
        # Lambda_rterm = Ezz
        Lambda_new = Lambda_lterm @ torch.inverse(Ezz)

        # Compute Psi_diag_new (Equation 6, G&H 1996). Must use Lambda_new!
        Psi_rterm = linalg.sum_outers(y, y) - linalg.sum_outers(Lambda_new @ Ez, y)
        Psi_diag_new = 1. / n * torch.diag(Psi_rterm)

        return Lambda_new, Psi_diag_new

    def E_z_given_y(self, L, PLL_inv, y):
        """Computes the first moment (mean) of the latent variable z.
        See Equation 2 in Ghahramani and Hinton (1996).
        :param L:       Current Lambda.
        :param PLL_inv: The inverse of (Psi + Lambda Lambda^T).
        :param y:       Observations of shape (n_dimensions, n_features).
        :return:        Expectation of z given y.
        """
        assert len(PLL_inv.shape) == 2
        beta = L.t() @ PLL_inv
        return beta @ y

    def E_zzT_given_y(self, L, PLL_inv, y, k):
        """Computes the second moment (variance) of the latent variable z.
        See Equation 4 in Ghahramani and Hinton (1996).
        :param L:       Current Lambda.
        :param PLL_inv: The inverse of (Psi + Lambda Lambda^T).
        :param y:       Observations of shape (n_dimensions, n_features).
        :param k:       Dimension of latent variable z.
        :return:        Expectation of the variance of z given y.
        """
        _, n = y.shape
        assert len(PLL_inv.shape) == 2
        beta = L.t() @ PLL_inv
        I    = torch.eye(k, device=device)
        bL   = beta @ L
        by   = beta @ y
        byyb = torch.einsum('ib,ob->io', [by, by])
        return n * (I - bL) + byyb

    def estimate_z_given_y(self, y):
        """Computes the first moment (mean) of the latent variable z using
        fitted parameters.
        :param y: Observations of shape (n_features, n_samples).
        :return:  Estimated expectation of z (n_features, n_samples).
        """
        k = 3 * self.latent_dim
        Lambda, Psi_diag = self.tile_params()

        PLL_inv = linalg.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)
        z = self.E_z_given_y(Lambda, PLL_inv, y)
        return z

def neg_log_likelihood(self, y, Lambda=None, Psi_diag=None):
        """Appendix A (p. 5) in Ghahramani and Hinton (1996).
        :param y:        (n x p)-dimensional observations.
        :param Lambda:   Current value for Lambda parameter.
        :param Psi_diag: Current value for Psi parameter.
        :return:         The negative log likelihood of the parameters given y.
        """
        p, n = y.shape
        k = 3 * self.latent_dim

        if Lambda is None and Psi_diag is None:
            Lambda, Psi_diag = self.tile_params()

        PLL_inv = linalg.woodbury_inv(Psi_diag, Lambda, Lambda.t(), k)
        Ez  = self.E_z_given_y(Lambda, PLL_inv, y).t()
        Ezz = self.E_zzT_given_y(Lambda, PLL_inv, y, k).t()

        inv_Psi = torch.diag(LA.diag_inv(Psi_diag))

        A = 1/2. * torch.diag(y.t() @ inv_Psi @ y)
        B = torch.diag(y.t() @ inv_Psi @ Lambda @ Ez.t())
        C = 1/2. * torch.trace(Lambda.t() @ inv_Psi @ Lambda @ Ezz)
        rterm_sum = (A - B).sum() + C

        # This computes: `logdet = -n/2. * log(det(diag(Psi_diag)))`.
        logdet = -n/2. * log(Psi_diag).sum()

        ll = (logdet - rterm_sum).item()
        nll = -ll
        return nll


    def reparameterize(self, Lambda, Psi_diag, z):
        """Performs the reparameterization trick for a Gaussian random variable.
        For details, see:
            http://blog.shakirm.com/2015/10/
            machine-learning-trick-of-the-day-4-reparameterisation-tricks/
        :param Lambda:   Current value for Lambda parameter.
        :param Psi_diag: Current value for Psi parameter.
        :param z:        Latent variable.
        :return:         Samples of y from estimated parameters Lambda and Psi.
        """
        n = z.shape[1]
        p = Psi_diag.shape[0]
        eps = torch.randn(p, n, device=cuda.device())
        # For numerical stability. For Psi to be PD, all elements must be
        # positive: https://math.stackexchange.com/a/927916/159872.
        Psi_diag = linalg.to_positive(Psi_diag)
        R = torch.cholesky(diag(Psi_diag), upper=False)
        return Lambda @ z + R @ eps

    def init_params(self):
        """Create model parameters and move them to appropriate device.
        :return: Model parameters.
        """
        p1 = self.p1
        p2 = self.p2
        k  = self.latent_dim

        Lambda1 = torch.randn(p1, k).to(device)
        Lambda2 = torch.randn(p2, k).to(device)

        B1 = torch.randn(p1, k).to(device)
        B2 = torch.randn(p2, k).to(device)

        log_Psi1_diag = torch.ones(p1).to(device)
        log_Psi2_diag = torch.ones(p2).to(device)

        return Lambda1, Lambda2, B1, B2, log_Psi1_diag, log_Psi2_diag


    def tile_params(self):
        """Tile parameters so we can use factor analysis updates for PCCA.
        :return: Model parameters concatenated appropriately.
        """
        p1 = self.p1
        p2 = self.p2
        k  = self.latent_dim

        B12 = torch.zeros(p1, k).to(device)
        B21 = torch.zeros(p2, k).to(device)

        Lambda = torch.cat((
            torch.cat((self.Lambda1, self.B1, B12), dim=1),
            torch.cat((self.Lambda2, B21, self.B2), dim=1)
        ), dim=0)

        log_Psi_diag = torch.cat((self.log_Psi1_diag, self.log_Psi2_diag))
        Psi_diag     = torch.exp(log_Psi_diag)

        return Lambda, Psi_diag

    def untile_params(self, Lambda, log_Psi_diag):
        """Takes tiled parameters and untiles them; reverse of `tile_params()`.
        :return: Model parameters unrolled or un-concatenated appropriately.
        """
        k = self.latent_dim
        Lambda1 = Lambda[:self.p1, :k]
        Lambda2 = Lambda[self.p1:, :k]
        log_Psi1_diag = log_Psi_diag[:self.p1]
        log_Psi2_diag = log_Psi_diag[self.p1:]

        B1 = Lambda[:self.p1, k:2 * k]
        B2 = Lambda[self.p1:, 2 * k:]
        return Lambda1, Lambda2, B1, B2, log_Psi1_diag, log_Psi2_diag


    # PyTorch will place a method `parameters` on our class. Don't step on it.
    def parameters_(self, view=None):
        """Generates an iterator of parameters for the PCCA instance.
        :param view: An optional string specifying which data type's
                     parameters to return.
        :return:     The aforementioned iterator.
        """
        if not view:
            params = [self.Lambda1, self.Lambda2, self.B1, self.B2,
                      self.log_Psi1_diag, self.log_Psi2_diag]
        elif view == 'y1':
            params = [self.Lambda1, self.B1]
        elif view == 'y2':
            params = [self.Lambda2, self.B2]
        else:
            raise AttributeError('Supported views: "y1", "y2".')

        return iter(params)