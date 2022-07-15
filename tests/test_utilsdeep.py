import numpy as np
import itertools as it
import os
from multiae import mcVAE, MVTCAE, MVAE, mmVAE, DVCCA, AAE, jointAAE, wAAE, AE


def test_mcVAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    multi_vae = mcVAE(input_dims=[20, 20]).to(DEVICE)
    multi_vae.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = multi_vae.predict_latents(test_1, test_2)
    recon = multi_vae.predict_reconstruction(test_1, test_2)


def test_jointVAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = MVAE(input_dims=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)


def test_MVTCAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    mvtcae = MVTCAE(input_dims=[20, 20]).to(DEVICE)
    mvtcae.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = mvtcae.predict_latents(test_1, test_2)
    recon = mvtcae.predict_reconstruction(test_1, test_2)


def test_mmVAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = mmVAE(input_dims=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)


def test_DVCCA():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = DVCCA(input_dims=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)


def test_AAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = AAE(input_dims=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)


def test_jointAAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = jointAAE(input_dims=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)


def test_wAAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = wAAE(input_dims=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)


def test_AE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = AE(input_dims=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)
