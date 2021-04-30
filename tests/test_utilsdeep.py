from src.models import utils_deep
from src.utils.io_utils import ConfigReader
import numpy as np
import itertools as it


def test_VAE():
    from src.models.vae import VAE
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    config_file = ConfigReader('./tests/test_config.yaml')
    DEVICE = 'cpu'
    parameters = {'sparse': [True, False], 'batch_size': [None, 10]}
    param_combs = list(it.product(*(parameters[key] for key in parameters)))
    config = config_file._conf
    for comb in param_combs:
        params = {key: comb[idx] for idx, key in enumerate(parameters)}
        for key,val in params.items():
            config[key] = val           
        models = VAE(input_dims=[20, 20], config=config).to(DEVICE)
        models.fit(train_1, train_2)
        latent_1, latent_2 = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)
def test_jointVAE(): 
    from src.models.joint_vae import VAE
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    config_file = ConfigReader('./tests/test_config.yaml')
    DEVICE = 'cpu'
    parameters = {'sparse': [True, False], 'batch_size': [None, 10]}   
    param_combs = list(it.product(*(parameters[key] for key in parameters)))
    config = config_file._conf
    for comb in param_combs:
        params = {key: comb[idx] for idx, key in enumerate(parameters)}
        for key,val in params.items():
            config[key] = val  
        models = VAE(input_dims=[20, 20], config=config).to(DEVICE)
        models.fit(train_1, train_2)
        latent = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)