import numpy as np
import itertools as it
import os
from multiae import mcVAE, MVTCAE
def test_mcVAE(): 
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    multi_vae = mcVAE(input_dims=[20, 20], n_epochs=10, batch_size=None).to(DEVICE)
    multi_vae.fit(train_1, train_2)
    latent = multi_vae.predict_latents(test_1, test_2)
    recon = multi_vae.predict_reconstruction(test_1, test_2)

def test_multiMVTCAE(): 
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    mvtcae = MVTCAE(input_dims=[20, 20], n_epochs=10, batch_size=50).to(DEVICE)
    mvtcae.fit(train_1, train_2)
    latent = mvtcae.predict_latents(test_1, test_2)
    recon = mvtcae.predict_reconstruction(test_1, test_2)

