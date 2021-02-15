import sys
import torch
from torchvision import datasets, transforms
import os
import numpy as np
from os.path import join, exists
import pandas as pd
from models.vae import VAE
from models.dvcca import DVCCA
from models.ae import AE
from utils.io_utils import ConfigReader, ResultsWriter
import numpy as np

if __name__ =='__main__':

    torch.manual_seed(42)  
    torch.cuda.manual_seed(42)
    config_file = ConfigReader('./src/config_files/config.yaml')

    MNIST_1 = datasets.MNIST('../data', train=True, transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    MNIST_2 = datasets.MNIST('../data', train=True, transform=transforms.Compose([
            transforms.RandomRotation((180,180)),
            transforms.ToTensor()
        ]))
    data_1 = MNIST_1.train_data.view(-1, 784).float()
    target = MNIST_1.train_labels
    data_2 = MNIST_2.train_data.view(-1, 784).float()

    models = VAE(input_dims=[784, 784], config=config_file._conf)
    logger = models.fit(data_1, data_2)

    MNIST_1 = datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    MNIST_2 = datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.RandomRotation((180,180)),
            transforms.ToTensor()
        ]))
    data_1 = MNIST_1.test_data.view(-1, 784).float()
    target = MNIST_1.test_labels
    data_2 = MNIST_2.test_data.view(-1, 784).float()

    latents = models.predict_latents(data_1,data_2)
    models.plot_UMAP(data=latents, target=target, title=['Original MNIST latent space', 'Rotated MNIST latent space'], title_short='MNIST')