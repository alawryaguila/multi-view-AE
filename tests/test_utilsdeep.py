from src.models import utils_deep
from src.utils.io_utils import ConfigReader
import numpy as np
def lets_check_this_works():
    pass

def test_VAE():
    from src.models.vae import VAE
    DEVICE = 'cpu'
    config_file = ConfigReader('./test_config.yaml')
    view_1 = np.random.rand(200, 20)
    view_2 = np.random.rand(200, 20)
    models = VAE(input_dims=[20, 20], config=config_file._conf).to(DEVICE)
    models.fit()