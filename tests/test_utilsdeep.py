import numpy as np
import itertools as it
import os

def test_VAE():
    from multiae import multiVAE
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    parameters = {'threshold': [0, 0.2], 'batch_size': [None,10]}
    param_combs = list(it.product(*(parameters[key] for key in parameters)))
    for comb in param_combs:
        path = str(os.getcwd()) + '/mcVAE/' + "_".join(str(item) for item in comb)
        params = {key: comb[idx] for idx, key in enumerate(parameters)}       
        models = multiVAE(input_dims=[20, 20], n_epochs=10, **params).to(DEVICE)
        models.fit(train_1, train_2, output_path=path)
        latent_1, latent_2 = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)

def test_multiVAE(): 
    from multiae import multiVAE
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    parameters = {'threshold': [0, 0.2], 'batch_size': [None, 10]}   
    param_combs = list(it.product(*(parameters[key] for key in parameters)))
    for comb in param_combs:
        path = str(os.getcwd()) + '/multiVAE/' + "_".join(str(item) for item in comb)
        params = {key: comb[idx] for idx, key in enumerate(parameters)}
        models = multiVAE(input_dims=[20, 20], n_epochs=10, **params).to(DEVICE)
        models.fit(train_1, train_2, output_path=path)
        latent = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)

def test_DVCCA(): 
    from multiae import DVCCA
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    parameters = {'z_dim': [5, 10], 'batch_size': [None, 10]}   
    param_combs = list(it.product(*(parameters[key] for key in parameters)))
    for comb in param_combs:
        path = str(os.getcwd()) + '/DVCCA/' + "_".join(str(item) for item in comb)
        params = {key: comb[idx] for idx, key in enumerate(parameters)}
        models = DVCCA(input_dims=[20, 20], n_epochs=10, **params).to(DEVICE)
        models.fit(train_1, train_2, output_path=path)
        latent = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)

def test_jointVAE(): 
    from multiae import jointVAE
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    parameters = {'z_dim': [5, 10], 'batch_size': [None, 10]}   
    param_combs = list(it.product(*(parameters[key] for key in parameters)))
    for comb in param_combs:
        path = str(os.getcwd()) + '/jointVAE/' + "_".join(str(item) for item in comb)
        params = {key: comb[idx] for idx, key in enumerate(parameters)}
        models = jointVAE(input_dims=[20, 20], n_epochs=10, **params).to(DEVICE)
        models.fit(train_1, train_2, output_path=path)
        latent = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)

def test_mmVAE(): 
    from multiae import mmVAE
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    parameters = {'z_dim': [5, 10], 'batch_size': [None, 10]}   
    param_combs = list(it.product(*(parameters[key] for key in parameters)))
    for comb in param_combs:
        path = str(os.getcwd()) + '/mmVAE/' + "_".join(str(item) for item in comb)
        params = {key: comb[idx] for idx, key in enumerate(parameters)}
        models = mmVAE(input_dims=[20, 20], n_epochs=10, **params).to(DEVICE)
        models.fit(train_1, train_2, output_path=path)
        latent = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)

def test_MVTCAE(): 
    from multiae import MVTCAE
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    parameters = {'z_dim': [5, 10], 'batch_size': [None, 10]}   
    param_combs = list(it.product(*(parameters[key] for key in parameters)))
    for comb in param_combs:
        path = str(os.getcwd()) + '/MVTCAE/' + "_".join(str(item) for item in comb)
        params = {key: comb[idx] for idx, key in enumerate(parameters)}
        models = MVTCAE(input_dims=[20, 20], n_epochs=10, **params).to(DEVICE)
        models.fit(train_1, train_2, output_path=path)
        latent = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)

def test_AAE(): 
    from multiae import AAE
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    parameters = {'z_dim': [5, 10], 'batch_size': [None, 10]}   
    param_combs = list(it.product(*(parameters[key] for key in parameters)))   
    for comb in param_combs:
        path = str(os.getcwd()) + '/AAE/' + "_".join(str(item) for item in comb)
        params = {key: comb[idx] for idx, key in enumerate(parameters)}
        models = AAE(input_dims=[20, 20], n_epochs=10, **params).to(DEVICE)
        models.fit(train_1, train_2, output_path=path)
        latent = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)

def test_jointAAE(): 
    from multiae import jointAAE
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    parameters = {'z_dim': [5, 10], 'batch_size': [None, 10]}   
    param_combs = list(it.product(*(parameters[key] for key in parameters)))
    for comb in param_combs:
        path = str(os.getcwd()) + '/jointAAE/' + "_".join(str(item) for item in comb)
        params = {key: comb[idx] for idx, key in enumerate(parameters)}
        models = jointAAE(input_dims=[20, 20], n_epochs=10, **params).to(DEVICE)
        models.fit(train_1, train_2, output_path=path)
        latent = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)

def test_wAAE(): 
    from multiae import wAAE
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    parameters = {'z_dim': [5, 10], 'batch_size': [None, 10]}   
    param_combs = list(it.product(*(parameters[key] for key in parameters)))
    for comb in param_combs:
        path = str(os.getcwd()) + '/wAAE/' + "_".join(str(item) for item in comb)
        params = {key: comb[idx] for idx, key in enumerate(parameters)}
        models = wAAE(input_dims=[20, 20], n_epochs=10, **params).to(DEVICE)
        models.fit(train_1, train_2, output_path=path)
        latent = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)

def test_AE(): 
    from multiae import AE
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    DEVICE = 'cpu'
    parameters = {'z_dim': [5, 10], 'batch_size': [None, 10]}   
    param_combs = list(it.product(*(parameters[key] for key in parameters)))
    for comb in param_combs:
        path = str(os.getcwd()) + '/AE/' + "_".join(str(item) for item in comb)
        params = {key: comb[idx] for idx, key in enumerate(parameters)}
        models = AE(input_dims=[20, 20], n_epochs=10, **params).to(DEVICE)
        models.fit(train_1, train_2, output_path=path)
        latent = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)

def test_classiferVAE(): 
    from multiae import VAE_classifier
    train_1 = np.random.rand(200, 20)
    train_2 = np.random.rand(200, 20)
    test_1 = np.random.rand(50, 20)
    test_2 = np.random.rand(50, 20)
    train_labels = np.random.randint(2, size=200)
    DEVICE = 'cpu'
    parameters = {'hidden_layers': [[], [100,50,10]], 'batch_size': [None, 10]}   
    param_combs = list(it.product(*(parameters[key] for key in parameters)))
    path = str(os.getcwd()) + '/VAEclassifier'
    for comb in param_combs:
        params = {key: comb[idx] for idx, key in enumerate(parameters)}
        models = VAE_classifier(input_dims=[20], n_labels=2, n_epochs=10, **params).to(DEVICE)
        models.fit(train_1, train_2, labels=train_labels, output_path=path)
        latent = models.predict_latents(test_1, test_2)
        recon = models.predict_reconstruction(test_1, test_2)
        pred = models.predict_labels(test_1, test_2)

