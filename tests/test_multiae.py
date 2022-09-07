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
    twoview_models = TWOVIEW_MODELS
    module = importlib.import_module("multiae")
    for m in test_models:
        class_ = getattr(module, m)
        if m not in twoview_models:
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
            "./user_config/example_dvcca.yaml" : [MODEL_DVCCA],
            "./user_config/example_sparse.yaml" : SPARSE_MODELS,
            "./user_config/example_multivariatenormal.yaml": VARIATIONAL_MODELS,
            "./user_config/example_multivariatenormal.yaml": SPARSE_MODELS,
            }
            
    module = importlib.import_module("multiae")
    train_twoviews = [train_1, train_2]
    test_twoviews = [test_1, test_2]
    train_threeviews = [train_1, train_2, train_3]
    test_threeviews = [test_1, test_2, test_3]

    for cfg, test_models in tests.items():
        for m in test_models:
            if m in TWOVIEW_MODELS:
                train = train_twoviews
                test = test_twoviews
            else:
                train = train_threeviews
                test = test_threeviews
            class_ = getattr(module, m)
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
        transforms.ToTensor()
    ]))
    MNIST_2 = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    data_1 = MNIST_1.train_data.reshape(-1, 784).float() / 255.
    target = MNIST_1.train_labels
    data_2 = MNIST_2.train_data.float()
    data_2 = torch.rot90(data_2, 1, [1, 2])
    data_2 = data_2.reshape(-1,784)/255.

    MNIST_1 = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    MNIST_2 = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))

    data_test_1 = MNIST_1.test_data.reshape(-1, 784).float() / 255.
    target_test = MNIST_1.test_labels.numpy()
    data_test_2 = MNIST_2.test_data.float() / 255.
    data_test_2 = torch.rot90(data_test_2, 1, [1, 2])
    data_test_2 = data_test_2.reshape(-1,784)

    cfg = "./user_config/example_mnist.yaml"
    test_models = [MODEL_MCVAE, MODEL_DVCCA]
    max_epochs = 10
    batch_size = 2000

    module = importlib.import_module("multiae")
    for m in test_models:
        class_ = getattr(module, m)
        model = class_(cfg=abspath(join(dirname( __file__ ), cfg)), input_dim=[784,784])

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

if __name__ == "__main__":
    test_models()
    test_userconfig()
    test_mnist()
