import numpy as np
import itertools as it
import os
from multiae import AE, AAE, jointAAE, wAAE, mcVAE, mVAE, mmVAE, mvtCAE, DVCCA


def test_mcVAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    multi_vae = mcVAE(input_dim=[20, 20]).to(DEVICE)
    multi_vae.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = multi_vae.predict_latents(test_1, test_2)
    recon = multi_vae.predict_reconstruction(test_1, test_2)


def test_jointVAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = mVAE(input_dim=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)


def test_MVTCAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    mvtcae = mvtCAE(input_dim=[20, 20]).to(DEVICE)
    mvtcae.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = mvtcae.predict_latents(test_1, test_2)
    recon = mvtcae.predict_reconstruction(test_1, test_2)


def test_mmVAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = mmVAE(input_dim=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)


def test_DVCCA():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = DVCCA(input_dim=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)


def test_AAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = AAE(input_dim=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)


def test_jointAAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = jointAAE(input_dim=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)


def test_wAAE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = wAAE(input_dim=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)


def test_AE():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = "cpu"
    model = AE(input_dim=[20, 20]).to(DEVICE)
    model.fit(train_1, train_2, max_epochs=5, batch_size=50)
    latent = model.predict_latents(test_1, test_2)
    recon = model.predict_reconstruction(test_1, test_2)

test_AE()

test_AAE()
test_jointAAE()
test_wAAE()

test_mcVAE()
test_jointVAE()
test_mmVAE()
test_MVTCAE()
test_DVCCA()
