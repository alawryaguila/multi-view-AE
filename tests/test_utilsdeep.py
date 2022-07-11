import numpy as np
import itertools as it
import os
from multiae import mcVAE, MVTCAE, MVAE, mmVAE, DVCCA

def test_mcVAE(): 
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    multi_vae = mcVAE(input_dims=[20, 20]).to(DEVICE)
    multi_vae.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = multi_vae.predict_latents(test_1, test_2)
    recon = multi_vae.predict_reconstruction(test_1, test_2)

def test_jointVAE(): 
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    models = MVAE(input_dims=[20, 20]).to(DEVICE)
    models.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = models.predict_latents(test_1, test_2)
    recon = models.predict_reconstruction(test_1, test_2)


def test_multiMVTCAE(): 
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    mvtcae = MVTCAE(input_dims=[20, 20]).to(DEVICE)
    mvtcae.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = mvtcae.predict_latents(test_1, test_2)
    recon = mvtcae.predict_reconstruction(test_1, test_2)

def test_mmVAE(): 
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    models = mmVAE(input_dims=[20, 20]).to(DEVICE)
    models.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = models.predict_latents(test_1, test_2)
    recon = models.predict_reconstruction(test_1, test_2)


def test_DVCCA(): 
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    models = DVCCA(input_dims=[20, 20]).to(DEVICE)
    models.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = models.predict_latents(test_1, test_2)
    recon = models.predict_reconstruction(test_1, test_2)