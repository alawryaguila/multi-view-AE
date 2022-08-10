import os
import numpy as np

from multiae import AE
from multiae import AAE
from multiae import jointAAE
from multiae import wAAE
from multiae import mcVAE
from multiae import mVAE
from multiae import mmVAE
from multiae import mvtCAE
from multiae import DVCCA

def test_AE(cfg=None):
    print("Testing: AE")
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    # model = AE(input_dim=[20]) # TODO: does not work, optimizer_idx in trainer_step error..
    # model.fit(train_1)

    model = AE(cfg=cfg, input_dim=[20, 10, 5])

    model.fit(train_1, train_2, train_3)
    model.fit(train_1, train_2, train_3, max_epochs=5, batch_size=5)

    latent = model.predict_latents(test_1, test_2, test_3)
    recon = model.predict_reconstruction(test_1, test_2, test_3)

    latent = model.predict_latents(test_1, test_2, test_3, batch_size=10)
    recon = model.predict_reconstruction(test_1, test_2, test_3, batch_size=5)   # TODO: doesn't work with batch_size

def test_AAE(cfg=None):
    print("Testing: AAE")
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    model = AAE(cfg=cfg, input_dim=[20, 10, 5])

    model.fit(train_1, train_2, train_3)
    model.fit(train_1, train_2, train_3, max_epochs=5, batch_size=5)

    latent = model.predict_latents(test_1, test_2, test_3)
    recon = model.predict_reconstruction(test_1, test_2, test_3)

    latent = model.predict_latents(test_1, test_2, test_3, batch_size=10)
    # recon = model.predict_reconstruction(test_1, test_2, test_3, batch_size=5)   # TODO: doesn't work with batch_size

def test_jointAAE(cfg=None):
    print("Testing: jointAAE")
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    model = jointAAE(cfg=cfg, input_dim=[20, 10, 5])

    model.fit(train_1, train_2, train_3)
    model.fit(train_1, train_2, train_3, max_epochs=5, batch_size=5)

    latent = model.predict_latents(test_1, test_2, test_3)
    recon = model.predict_reconstruction(test_1, test_2, test_3)

    latent = model.predict_latents(test_1, test_2, test_3, batch_size=10)
    # recon = model.predict_reconstruction(test_1, test_2, test_3, batch_size=5)   # TODO: doesn't work with batch_size

def test_wAAE(cfg=None):
    print("Testing: wAAE")
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    model = wAAE(cfg=cfg, input_dim=[20, 10, 5])

    model.fit(train_1, train_2, train_3)
    model.fit(train_1, train_2, train_3, max_epochs=5, batch_size=5)

    latent = model.predict_latents(test_1, test_2, test_3)
    recon = model.predict_reconstruction(test_1, test_2, test_3)

    latent = model.predict_latents(test_1, test_2, test_3, batch_size=10)
    # recon = model.predict_reconstruction(test_1, test_2, test_3, batch_size=5)   # TODO: doesn't work with batch_size

def test_mcVAE(cfg=None):
    print("Testing: mcVAE")
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    model = mcVAE(cfg=cfg, input_dim=[20, 10, 5])

    model.fit(train_1, train_2, train_3)
    model.fit(train_1, train_2, train_3, max_epochs=5, batch_size=5)

    latent = model.predict_latents(test_1, test_2, test_3)
    recon = model.predict_reconstruction(test_1, test_2, test_3)  # TODO: has different shape than predict_latent

    latent = model.predict_latents(test_1, test_2, test_3, batch_size=10)
    # recon = model.predict_reconstruction(test_1, test_2, test_3, batch_size=5)   # TODO: doesn't work with batch_size

def test_mVAE(cfg=None):
    print("Testing: mVAE")
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    model = mVAE(cfg=cfg, input_dim=[20, 10, 5])

    model.fit(train_1, train_2, train_3)
    model.fit(train_1, train_2, train_3, max_epochs=5, batch_size=5)

    latent = model.predict_latents(test_1, test_2, test_3)
    recon = model.predict_reconstruction(test_1, test_2, test_3)  # TODO: has different shape than predict_latent

    latent = model.predict_latents(test_1, test_2, test_3, batch_size=10)
    # recon = model.predict_reconstruction(test_1, test_2, test_3, batch_size=5)   # TODO: doesn't work with batch_size

def test_mmVAE(cfg=None):
    print("Testing: mmVAE")
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    model = mmVAE(cfg=cfg, input_dim=[20, 10, 5])

    model.fit(train_1, train_2, train_3)
    model.fit(train_1, train_2, train_3, max_epochs=5, batch_size=5)

    latent = model.predict_latents(test_1, test_2, test_3)
    recon = model.predict_reconstruction(test_1, test_2, test_3)  # TODO: has different shape than predict_latent

    latent = model.predict_latents(test_1, test_2, test_3, batch_size=10)
    # recon = model.predict_reconstruction(test_1, test_2, test_3, batch_size=5)   # TODO: doesn't work with batch_size

def test_mvtCAE(cfg=None):
    print("Testing: mvtCAE")
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    model = mvtCAE(cfg=cfg, input_dim=[20, 10, 5])

    model.fit(train_1, train_2, train_3)
    model.fit(train_1, train_2, train_3, max_epochs=5, batch_size=5)

    latent = model.predict_latents(test_1, test_2, test_3)
    recon = model.predict_reconstruction(test_1, test_2, test_3)  # TODO: has different shape than predict_latent

    latent = model.predict_latents(test_1, test_2, test_3, batch_size=10)
    # recon = model.predict_reconstruction(test_1, test_2, test_3, batch_size=5)   # TODO: doesn't work with batch_size

def test_DVCCA(cfg=None):
    print("Testing: DVCCA")
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    # model = DVCCA(input_dim=[20, 10, 5])
    model = DVCCA(cfg=cfg, input_dim=[20, 10, 5])

    model.fit(train_1, train_2, train_3)
    model.fit(train_1, train_2, train_3, max_epochs=5, batch_size=5)

    latent = model.predict_latents(test_1, test_2, test_3)
    recon = model.predict_reconstruction(test_1, test_2, test_3)  # TODO: has different shape than predict_latent

    latent = model.predict_latents(test_1, test_2, test_3, batch_size=10)
    # recon = model.predict_reconstruction(test_1, test_2, test_3, batch_size=5)   # TODO: doesn't work with batch_size

def test_models():
    test_AE()
    test_AAE()
    test_jointAAE()
    test_wAAE()
    test_mcVAE()
    test_mVAE()
    test_mmVAE()
    test_mvtCAE()
    test_DVCCA()

def test_userconfig():
    test_DVCCA(cfg="/tests/user_config/sample_dvcca.yaml")

if __name__ == "__main__":
    test_models()
    test_userconfig()
