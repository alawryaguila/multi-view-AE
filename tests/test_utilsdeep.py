from src.models import utils_deep
from src.utils.io_utils import ConfigReader
import numpy as np
import itertools as it


class test_deepmodels:
    def test_setup(self):
        self.train_1 = np.random.rand(200, 20)
        self.train_2 = np.random.rand(200, 20)
        self.test_1 = np.random.rand(50, 20)
        self.test_2 = np.random.rand(50, 20)
        self.config_file = ConfigReader('./tests/test_config.yaml')
        self.DEVICE = 'cpu'
        self.parameters = {'sparse': [True, False], 'batch_size': [None, 10]}
    def test_VAE(self):
        from src.models.vae import VAE
        param_combs = list(it.product(*(self.parameters[key] for key in self.parameters)))
        config = self.config_file._conf
        for comb in parameter_combs:
            for key,val in comb.items():
                config[key] = val           
            models = VAE(input_dims=[20, 20], config=config).to(self.DEVICE)
            models.fit(self.train_1, self.train_2)
            latent_1, latent_2 = models.predict_latents(self.test_1, self.test_2)
            recon = models.predict_reconstruction(self.test_1, self.test_2)
    def test_jointVAE(self):
        from src.models.joint_vae import VAE
        param_combs = list(it.product(*(self.parameters[key] for key in self.parameters)))
        config = self.config_file._conf
        for comb in parameter_combs:
            for key,val in comb.items():
                config[key] = val  
            models = VAE(input_dims=[20, 20], config=config).to(self.DEVICE)
            models.fit(self.train_1, self.train_2)
            latent = models.predict_latents(self.test_1, self.test_2)
            recon = models.predict_reconstruction(self.test_1, self.test_2)