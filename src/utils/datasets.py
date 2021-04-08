'''
Class for loading data into Pytorch float tensor

From: https://gitlab.com/acasamitjana/latentmodels_ad

'''
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data, indices = False, transform=None):
        self.data = data
        if isinstance(data,list):
            if isinstance(data,np.ndarray):
                self.data = [torch.from_numpy(d).float() for d in self.data ]
            self.N = len(self.data[0])
            self.shape = np.shape(self.data[0])
        else:
            if isinstance(data,np.ndarray):
                self.data = torch.from_numpy(self.data).float()
            self.N = len(self.data)
            self.shape = np.shape(self.data)

        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if isinstance(self.data,list):
            x = [d[index] for d in self.data]
        else:
            x = self.data[index]

        if self.transform:
            x = self.transform(x)

        if self.indices:
            return x, index
        else:
            return x

    def __len__(self):
        return self.N


def generate_simulated_data(m: int, k: int, N: int, M: int, sparse_variables_1: float = 0,
                            sparse_variables_2: float = 0,
                            signal: float = 1,
                            structure: str = 'identity', sigma: float = 0.9, decay: float = 0.5,
                            rand_eigs_1: bool = False,
                            rand_eigs_2: bool = False):
    """
    :param m: number of samples
    :param k: number of latent dimensions
    :param N: number of features in view 1
    :param M: number of features in view 2
    :param sparse_variables_1: fraction of active variables from view 1 associated with true signal
    :param sparse_variables_2: fraction of active variables from view 2 associated with true signal
    :param signal: correlation
    :param structure: within view covariance structure
    :param sigma: gaussian sigma
    :param decay: ratio of second signal to first signal
    :param rand_eigs_1:
    :param rand_eigs_2:
    :return: tuple of numpy arrays: view_1, view_2, true weights from view 1, true weights from view 2, overall covariance structure
    """
    mean = np.zeros(N + M)
    cov = np.zeros((N + M, N + M))
    p = np.arange(0, k)
    p = decay ** p
    # Covariance Bit
    if structure == 'identity':
        cov_1 = np.eye(N)
        cov_2 = np.eye(M)
    elif structure == 'gaussian':
        x = np.linspace(-1, 1, N)
        x_tile = np.tile(x, (N, 1))
        mu_tile = np.transpose(x_tile)
        dn = 2 / (N - 1)
        cov_1 = gaussian(x_tile, mu_tile, sigma, dn)
        cov_1 /= cov_1.max()
        x = np.linspace(-1, 1, M)
        x_tile = np.tile(x, (M, 1))
        mu_tile = np.transpose(x_tile)
        dn = 2 / (M - 1)
        cov_2 = gaussian(x_tile, mu_tile, sigma, dn)
        cov_2 /= cov_2.max()
    elif structure == 'toeplitz':
        c = np.arange(0, N)
        c = sigma ** c
        cov_1 = linalg.toeplitz(c, c)
        c = np.arange(0, M)
        c = sigma ** c
        cov_2 = linalg.toeplitz(c, c)
    elif structure == 'random':
        cov_1 = np.random.rand(N, N)
        U, S, V = np.linalg.svd(cov_1.T @ cov_1)
        cov_1 = U @ (1.0 + np.diag(np.random.rand(N))) @ V
        cov_2 = np.random.rand(M, M)
        U, S, V = np.linalg.svd(cov_2.T @ cov_2)
        cov_2 = U @ (1.0 + np.diag(np.random.rand(M))) @ V
    cov[:N, :N] = cov_1
    cov[N:, N:] = cov_2
    del cov_1
    del cov_2
    up = np.random.rand(N, k) - 0.5
    for _ in range(k):
        if sparse_variables_1 > 0:
            if sparse_variables_1 < 1:
                sparse_variables_1 = np.ceil(sparse_variables_1 * N).astype('int')
            first = np.random.randint(N - sparse_variables_1)
            up[:first, _] = 0
            up[(first + sparse_variables_1):, _] = 0
    up = decorrelate_dims(up, cov[:N, :N])
    up /= np.sqrt(np.diag((up.T @ cov[:N, :N] @ up)))
    vp = np.random.rand(M, k) - 0.5
    for _ in range(k):
        if sparse_variables_2 > 0:
            if sparse_variables_2 < 1:
                sparse_variables_2 = np.ceil(sparse_variables_2 * M).astype('int')
            first = np.random.randint(M - sparse_variables_2)
            vp[:first, _] = 0
            vp[(first + sparse_variables_2):, _] = 0
    vp = decorrelate_dims(vp, cov[N:, N:])
    vp /= np.sqrt(np.diag((vp.T @ cov[N:, N:] @ vp)))
    cross = np.zeros((N, M))
    for _ in range(k):
        cross += signal * p[_] * np.outer(up[:, _], vp[:, _])
    # Cross Bit
    cross = cov[:N, :N] @ cross @ cov[N:, N:]
    cov[N:, :N] = cross.T
    cov[:N, N:] = cross
    if cov.shape[0] < 2000:
        X = np.random.multivariate_normal(mean, cov, m)
    else:
        X = np.zeros((m, N + M))
        chol = np.linalg.cholesky(cov)
        for _ in range(m):
            X[_, :] = chol_sample(mean, chol)
    Y = X[:, N:]
    X = X[:, :N]
    return X, Y, up, vp, cov


class GeneratorUniform(torch.nn.Module):
    
	def __init__(
			self,
			lat_dim=2,
			n_channels=2,
			n_feats=5,
			seed=100,
	):
		"""
		Generate multiple sources (channels) of data through a linear generative model:

		z ~ N(0,I)

		for ch in N_channels:
			x_ch = W_ch(z)

		where:

			"W_ch" is an arbitrary linear mapping z -> x_ch

		:param lat_dim:
		:param n_channels:
		:param n_feats:
		"""
		super().__init__()

		self.lat_dim = lat_dim
		self.n_channels = n_channels
		self.n_feats = n_feats

		self.seed = seed
		np.random.seed(self.seed)

		W = []

		for ch in range(n_channels):
			w_ = np.random.uniform(-1, 1, (self.n_feats, lat_dim)) #draws from uniform distribution
			u, s, vt = np.linalg.svd(w_, full_matrices=False) #take u or v to get orthogonal vectors
			w = u if self.n_feats >= lat_dim else vt
			W.append(torch.nn.Linear(lat_dim, self.n_feats, bias=False))
			W[ch].weight.data = torch.FloatTensor(w)

		self.W = torch.nn.ModuleList(W)

	def forward(self, z):

		if isinstance(z, list):
			return [self.forward(_) for _ in z]

		if type(z) == np.ndarray:
			z = torch.FloatTensor(z)

		assert z.size(1) == self.lat_dim

		obs = []
		for ch in range(self.n_channels):
			x = self.W[ch](z)
			obs.append(x.detach())

		return obs

class GeneratorTesting(torch.nn.Module):
    
    def __init__(
            self,
            lat_dim=2,
            n_channels=2,
            n_feats=5,
            seed=100,
    ):
        """
        Generate multiple sources (channels) of data through a gaussian generative model:

        x ~ N(Wz,sigma)

        for ch in N_channels:
            x_ch = W_ch(z)

        where:

            "W_ch" is an arbitrary linear mapping z -> x_ch

        :param lat_dim:
        :param n_channels:
        :param n_feats:
        """
        super().__init__()

        self.lat_dim = lat_dim
        self.n_channels = n_channels
        self.n_feats = n_feats

        self.seed = seed
        np.random.seed(self.seed)
    def forward(self, z):
        X = []
        W = []
        Sigma = []
        for ch in range(self.n_channels):
            w_ = np.random.uniform(-1,1,(self.n_feats, self.lat_dim))
            u, s, vt = np.linalg.svd(w_, full_matrices=False) #do I need this bit?
            w = u if self.n_feats >= self.lat_dim else vt #this bit makes the cov matrix positive semidefinite
            wz = np.dot(z, np.transpose(w))
            W.append(w)
            sigma = np.identity(self.n_feats) - np.dot(w, np.transpose(w))
            Sigma.append(sigma)
            x_ = wz + np.random.multivariate_normal(np.zeros(self.n_feats), sigma, np.shape(z)[0])
            #print(np.shape(x_))
            X.append(x_)
        self.W = W
        self.Sigma = Sigma
        return X


class SyntheticDataset(Dataset): #this is just for x_c = G_c*z. the noise is added later
    
    def __init__(
            self,
            n=500,
            lat_dim=2,
            n_feats=5,
            n_channels=2,
            generatorclass=GeneratorTesting,
            train=True,
    ):
        self.n = n  # N subjects
        self.lat_dim = lat_dim
        self.n_feats = n_feats
        self.n_channels = n_channels
        self.train = train
        # here we define 2 Gaussian latents variables z = (l_1, l_2)
        seed = 7 if self.train is True else 14
        np.random.seed(seed)
        self.z = np.random.normal(size=(self.n, self.lat_dim))

        self.generator = generatorclass(lat_dim=self.lat_dim, n_channels=self.n_channels, n_feats=self.n_feats)

        self.x = self.generator(self.z)
        self.W = self.generator.W
        self.Sigma = self.generator.Sigma

    def __len__(self):

        return self.n

    def __getitem__(self, item):

        return [x[item] for x in self.x]

