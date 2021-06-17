'''
Class for loading data into Pytorch float tensor

From: https://gitlab.com/acasamitjana/latentmodels_ad

'''
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyDataset_SNPs(Dataset):
    def __init__(self, data):
        self.data
    def __getitem__(self, index):
        x = self.data.values[index]
        return x

    def __len__(self):
        return self.N

class MyDataset(Dataset):
    def __init__(self, data, indices = False, transform=None):
        self.data = data
        if isinstance(data,list):
            if isinstance(data,np.ndarray):
                self.data = [torch.from_numpy(d).float() for d in self.data]
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

class MyDataset_labels(Dataset):
    def __init__(self, data, labels, indices = False, transform=None):
        self.data = data
        self.labels = labels
        if isinstance(data,list):
            if isinstance(data,np.ndarray):
                self.data = [torch.from_numpy(d).float() for d in self.data]
                
            self.N = len(self.data[0])
            self.shape = np.shape(self.data[0])
        else:
            if isinstance(data,np.ndarray):
                self.data = torch.from_numpy(self.data).float()
            self.N = len(self.data)
            self.shape = np.shape(self.data)
        self.labels = torch.from_numpy(self.labels).float()
        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if isinstance(self.data,list):
            x = [d[index] for d in self.data]
        else:
            x = self.data[index]

        if self.transform:
            x = self.transform(x)
        t = self.labels[index]
        if self.indices:
            return x, t, index
        return x, t

    def __len__(self):
        return self.N

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
            x_ = wz + np.random.multivariate_normal(np.zeros(self.n_feats), sigma, np.shape(z)[0]) #what does this bit do?
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
            generatorclass=GeneratorUniform,
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
        #self.Sigma = self.generator.Sigma

    def __len__(self):

        return self.n

    def __getitem__(self, item):

        return [x[item] for x in self.x]

