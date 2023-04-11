import os
import numpy as np
import torch
import importlib

from multiviewae import *
from os.path import abspath, dirname, join
from torchvision import datasets, transforms

def print_results(key, res, idx=0):
    """Function to print the model results.

    Args:
        key (_type_): Description of result.
        res (_type_): Result to print.
        idx (int, optional): Result index. Defaults to 0.
    """
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
    """Train and test each model in the library using 1-3 views of simulated data and the default model parameters.
    The purpose of this test is to test the general functionality and user defined arguments of the 
    fit(), fit(), predict_latents(), and predict_reconstruction() methods. The test will fail if running the
    fit(), predict_latents(), and predict_reconstruction() methods fails using the data and arguments provided.
    """
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    test_models = MODELS
    module = importlib.import_module("multiviewae")
    for m in test_models:
        print('MODEL CLASS')
        print(m)
        class_ = getattr(module, m)
        if m not in [MODEL_JMVAE]: #JMVAE only designed for 2 views of data
            model1 = class_(input_dim=[20])
            model1.fit(train_1) #fit model with 1 view
            model1.fit(train_1, max_epochs=5, batch_size=10) #fit using user specified max_epochs and batch size

            model2 = class_(input_dim=[20, 10, 5])
            model2.fit(train_1, train_2, train_3) #fit model with 3 views
            model2.fit(train_1, train_2, train_3, max_epochs=5, batch_size=5)

            print("RESULTS: ", m)
            latent = model1.predict_latents(test_1) 
            print_results("latent", latent)
            recon = model1.predict_reconstruction(test_1)
            print_results("recon", recon)

            latent = model1.predict_latents(test_1, batch_size=10) #test user defined batch size
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
            model1.fit(train_1, train_2) #fit model with 2 views
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
    """Train and test each model in the library using 1-3 views of simulated data.
    The purpose of this test is to test the ability to add user defined configuration files altering various default model parameters.
    The test will fail if any of the settings provided in the configuration files (Laplace distribution, Multivariate normal distribution, sparsity constraint) 
    are not functioning correctly or if the configuration file reading and processing functionality is not working.
    """
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 10)
    train_3 = np.random.rand(200, 5)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 10)
    test_3 = np.random.rand(50, 5)

    tests = {
            "./user_config/dvcca.yaml" : [MODEL_DVCCA], #test private DVCCA
            "./user_config/mmjsd.yaml" : [MODEL_MMJSD], #test mmJSD without private latents
            "./user_config/laplace.yaml": VARIATIONAL_MODELS, #tests using laplace for decoding distribution
            "./user_config/sparse.yaml" : SPARSE_MODELS, #test sparse models
            "./user_config/multivariatenormal.yaml": VARIATIONAL_MODELS, #tests using multivariate normal for decoding distribution
            "./user_config/multivariatenormal.yaml": SPARSE_MODELS, #tests using multivariate normal for decoding distribution of sparse models

            }

    module = importlib.import_module("multiviewae")
    train_twoviews = [train_1, train_2]
    test_twoviews = [test_1, test_2]
    train_threeviews = [train_1, train_2, train_3]
    test_threeviews = [test_1, test_2, test_3]

    for cfg, test_models in tests.items(): #test all configuration files
        for m in test_models: #test all models for which configuration file is compatible
            class_ = getattr(module, m)
            if m in [MODEL_JMVAE, MODEL_DVCCA]: #JMVAE and DVCCA models are only designed for 2 views
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
    """Tests the MNIST example code. The test will fail if unable to train and test a subset of the multiviewae models with the MNIST example data.
    """
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

    module = importlib.import_module("multiviewae")
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
    """Tests the ability of the validation function to catch non-compatible parameter combinations provided in configuration files.
    The test will fail if any invalid parameter combinations are not flagged by config_schema. 
    """
    tests = {
            "./user_config/validation_decoder.yaml" : MODELS, #can't have variational decoder and bernoulli data distribution
            "./user_config/validation_adversarial1.yaml" : [MODEL_AE] + ADVERSARIAL_MODELS, #can't have variational encoder for adversarial or AE models
            "./user_config/validation_adversarial2.yaml" : [MODEL_AE] + ADVERSARIAL_MODELS, #can't have Normal encoding distribution for adversarial or AE models
            "./user_config/validation_variational1.yaml": VARIATIONAL_MODELS, #must use variational encoder for variational models
            "./user_config/validation_variational2.yaml": VARIATIONAL_MODELS, #encoding and prior distribution must be the same type
            "./user_config/validation_prior1.yaml": MODELS, #can't have different scale inputs for Normal prior distribution
            "./user_config/validation_prior2.yaml": VARIATIONAL_MODELS, #prior dimension and z dimension must be the same
            }

    module = importlib.import_module("multiviewae")

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

def test_architectures():
    """Test MLP and CNN functionality. This test tests the ability to specify different MLP architectures for different views, and
    the variational and non-variational CNN architecture functionality. The test will fail if the fit(), predict_latents() or predict_reconstruction()
    methods fail to run using a combination of MLP and CNN architectures for encoder and decoder layers specified in the user configuration files.
    """
    train_n = 200
    test_n = 50

    module = importlib.import_module("multiviewae")

    tests = {
            "" : [[10, 10], MODELS], #tests the default MLP architecture 
            "./user_config/mlp.yaml" : [[10, 10], [MODEL_AE] + ADVERSARIAL_MODELS], #specify different MLP architectures for 1st view to remaining views
            "./user_config/cnn.yaml" : [[(1, 32, 32), (1, 32,32)], [MODEL_AE] + ADVERSARIAL_MODELS], #test non-variational CNN encoder
            "./user_config/cnn_var.yaml" : [[(1, 32, 32), (1, 32,32)], [
                        MODEL_MCVAE,
                        MODEL_MVAE,
                        # MODEL_JMVAE, # does not support cnn
                        MODEL_MEMVAE,
                        # MODEL_MMVAE,  # currently does not support cnn
                        MODEL_MVTCAE,
                        MODEL_DVCCA,
                        MODEL_MOPOEVAE
                    ]] #test variational CNN encoder
            }

    module = importlib.import_module("multiviewae")
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
    test_architectures()
