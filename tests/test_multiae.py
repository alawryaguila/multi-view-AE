import os
import numpy as np
import torch
import importlib

from multiae import *
from os.path import abspath, dirname, join
from torchvision import datasets, transforms

def print_results(key, res, idx=0):
    if isinstance(res, list):
        x =" "*idx
        print(f"{x}{key}")
        for i, r in enumerate(res):
            print_results(i, r, idx+1)
    elif isinstance(res, (np.ndarray, torch.Tensor)):
        print(" "*idx, key, type(res), res.shape)
    else: # distributions
        x = res._sample()
        print(" "*idx, key, type(res), x.shape)


def test_models():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    test_models = MODELS
    module = importlib.import_module("multiae")
    for m in test_models:
        print('MODEL CLASS')
        print(m)
        class_ = getattr(module, m)
        if m not in [MODEL_JMVAE]:
            model1 = class_(input_dim=[20])
            model1.fit(train_1)
            model1.fit(train_1, max_epochs=5, batch_size=10)

            model2 = class_(input_dim=[20, 10, 5])
            model2.fit(train_1, train_2, train_3)
            model2.fit(train_1, train_2, train_3, max_epochs=5, batch_size=5)

            print("RESULTS: ", m)
            latent = model1.predict_latents(test_1)
            print_results("latent", latent)
            recon = model1.predict_reconstruction(test_1)
            print_results("recon", recon)

            latent = model1.predict_latents(test_1, batch_size=10)
            print_results("latent", latent)
            recon = model1.predict_reconstruction(test_1, batch_size=5)
            print_results("recon", recon)

            latent = model2.predict_latents(test_1, test_2, test_3)
            print_results("latent", latent)
            recon = model2.predict_reconstruction(test_1, test_2, test_3)
            print_results("recon", recon)

            latent = model2.predict_latents(test_1, test_2, test_3, batch_size=10)
            print_results("latent", latent)
            recon = model2.predict_reconstruction(test_1, test_2, test_3, batch_size=5)
            print_results("recon", recon)
            print("")
        else:
            model1 = class_(input_dim=[20, 10])
            model1.fit(train_1, train_2)
            model1.fit(train_1, train_2, max_epochs=5, batch_size=10)

            print("RESULTS: ", m)
            latent = model1.predict_latents(test_1, test_2)
            print_results("latent", latent)
            recon = model1.predict_reconstruction(test_1, test_2)
            print_results("recon", recon)

            latent = model1.predict_latents(test_1, test_2, batch_size=10)
            print_results("latent", latent)
            recon = model1.predict_reconstruction(test_1, test_2, batch_size=5)
            print_results("recon", recon)

def test_userconfig():
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    tests = {
            "./user_config/dvcca.yaml" : [MODEL_DVCCA],
            "./user_config/mmjsd.yaml" : [MODEL_MMJSD],
            "./user_config/laplace.yaml": VARIATIONAL_MODELS,
            "./user_config/sparse.yaml" : SPARSE_MODELS,
            "./user_config/multivariatenormal.yaml": VARIATIONAL_MODELS,
            "./user_config/multivariatenormal.yaml": SPARSE_MODELS,

            }

    module = importlib.import_module("multiae")
    train_twoviews = [train_1, train_2]
    test_twoviews = [test_1, test_2]
    train_threeviews = [train_1, train_2, train_3]
    test_threeviews = [test_1, test_2, test_3]
    input_dim = [20, 10, 5]

    for cfg, test_models in tests.items():
        for m in test_models:
            class_ = getattr(module, m)
            if m in [MODEL_JMVAE, MODEL_DVCCA]:
                train = train_twoviews
                test = test_twoviews
                model = class_(cfg=abspath(join(dirname( __file__ ), cfg)), input_dim=[20, 10])
            else:
                train = train_threeviews
                test = test_threeviews
                model = class_(cfg=abspath(join(dirname( __file__ ), cfg)), input_dim=[20, 10, 5])

            model.fit(*train)

            print("RESULTS: ", m)
            latent = model.predict_latents(*test)
            print_results("latent", latent)
            recon = model.predict_reconstruction(*test)
            print_results("recon", recon)

            latent = model.predict_latents(*test, batch_size=10)
            print_results("latent", latent)
            recon = model.predict_reconstruction(*test, batch_size=5)
            print_results("recon", recon)
            print("")

def test_mnist():
    MNIST_1 = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))



    data_1 = MNIST_1.train_data[:, :, :14].reshape(-1,392).float()/255.
    data_2 = MNIST_1.train_data[:, :, 14:].reshape(-1,392).float()/255.


    MNIST_1 = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ]))

    data_test_1 = MNIST_1.test_data[:, :, :14].reshape(-1,392).float()/255.
    data_test_2 = MNIST_1.test_data[:, :, 14:].reshape(-1,392).float()/255.


    cfg = "./user_config/mnist.yaml"
    test_models = [MODEL_MCVAE, MODEL_DVCCA, MODEL_MMJSD]
    max_epochs = 10
    batch_size = 2000

    module = importlib.import_module("multiae")
    for m in test_models:
        class_ = getattr(module, m)
        model = class_(cfg=abspath(join(dirname( __file__ ), cfg)), input_dim=[392,392])

        model.fit(data_1, data_2, max_epochs=max_epochs, batch_size=batch_size)

        print("RESULTS: ", m)
        latent = model.predict_latents(data_test_1, data_test_2)
        print_results("latent", latent)
        recon = model.predict_reconstruction(data_test_1, data_test_2)
        print_results("recon", recon)

        latent = model.predict_latents(data_test_1, data_test_2, batch_size=1000)
        print_results("latent", latent)
        recon = model.predict_reconstruction(data_test_1, data_test_2, batch_size=1000)
        print_results("recon", recon)
        print("")

def test_validation():

    tests = {
            "./user_config/validation_decoder.yaml" : MODELS,
            "./user_config/validation_adversarial1.yaml" : [MODEL_AE] + ADVERSARIAL_MODELS,
            "./user_config/validation_adversarial2.yaml" : [MODEL_AE] + ADVERSARIAL_MODELS,
            "./user_config/validation_variational1.yaml": VARIATIONAL_MODELS,
            "./user_config/validation_variational2.yaml": VARIATIONAL_MODELS,
            "./user_config/validation_prior1.yaml": MODELS,
            "./user_config/validation_prior2.yaml": VARIATIONAL_MODELS,
            }

    module = importlib.import_module("multiae")

    for cfg, test_models in tests.items():
        print(cfg, test_models)
        for m in test_models:
            class_ = getattr(module, m)
            try:
                model = class_(cfg=abspath(join(dirname( __file__ ), cfg)), input_dim=[20, 10])
            except Exception as e:
                print(f"Validation test OK: {m}\t{e}")
            else:
                print(f"Validation test NG: {m}")
                exit()
        print()

def test_cnn():
    train_n = 200
    test_n = 50

    module = importlib.import_module("multiae")

    tests = {
            "" : [[10, 10], MODELS],
            "./user_config/mlp.yaml" : [[10, 10], [MODEL_AE] + ADVERSARIAL_MODELS],
            "./user_config/cnn.yaml" : [[(1, 32, 32), (1, 32,32)], [MODEL_AE] + ADVERSARIAL_MODELS],
            "./user_config/cnn_var.yaml" : [[(1, 32, 32), (1, 32,32)], [
                        MODEL_MCVAE,
                        MODEL_MVAE,
                        # MODEL_JMVAE, # does not support cnn
                        MODEL_MEMVAE,
                        # MODEL_MMVAE,  # currently does not support cnn
                        MODEL_MVTCAE,
                        MODEL_DVCCA,
                        MODEL_MOPOEVAE
                    ]]
            }

    module = importlib.import_module("multiae")
    for cfg, [dim, models] in tests.items():
        train_data = []
        test_data = []
        for d in dim:
            if isinstance(d, int):
                train_data.append(np.random.rand(train_n, d))
                test_data.append(np.random.rand(test_n, d))
            else:
                train_data.append(np.random.rand(train_n, *d))
                test_data.append(np.random.rand(test_n, *d))

        for m in models:
            class_ = getattr(module, m)
            if len(cfg) != 0:
                model1 = class_(cfg=abspath(join(dirname( __file__ ), cfg)), input_dim=dim)
            else:
                model1 = class_(input_dim=dim)

            model1.fit(*train_data)
            model1.fit(*train_data, max_epochs=5, batch_size=10)

            print("RESULTS: ", m)
            latent = model1.predict_latents(*test_data)
            print_results("latent", latent)
            recon = model1.predict_reconstruction(*test_data)
            print_results("recon", recon)

            latent = model1.predict_latents(*test_data, batch_size=10)
            print_results("latent", latent)
            recon = model1.predict_reconstruction(*test_data, batch_size=5)
            print_results("recon", recon)


if __name__ == "__main__":
    test_models()
    test_userconfig()
    test_mnist()
    test_validation()
    test_cnn()
