import sys
import torch
from torchvision import datasets, transforms
import os
import numpy as np
from os.path import join, exists
import pandas as pd
from src.models.vae import VAE
from src.utils.io_utils import ConfigReader, ResultsWriter
import numpy as np

if __name__ =='__main__':
    DEVICE = torch.device("cuda")
    torch.manual_seed(42)  
    torch.cuda.manual_seed(42)
    config_file = ConfigReader('./src/config_files/config.yaml')

    MNIST_1 = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            #transforms.Normalize((0.1307,), (0.3081,)),
            transforms.ToTensor(),
            
        ]))
    MNIST_2 = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.RandomRotation((180,180)),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        ]))
    data_1 = MNIST_1.train_data.reshape(-1, 784).float() / 255.
    target = MNIST_1.train_labels
    data_2 = MNIST_2.train_data.reshape(-1, 784).float() / 255.

   # models = VAE(input_dims=[784], MNIST=True, **config_file._conf).to(DEVICE)

    trainer_dict = {
                    'checkpoint_metric_name': 'val_loss',
                    'checkpoint_monitor_mode': 'min',
                    'early_stopping': True,
                    'early_stopping_delta': 0.0001,
                    'early_stopping_patience': 30, 
                    }
    models = VAE(input_dims=[784], dist='bernoulli', trainer_dict=trainer_dict,**config_file._conf)
    path='../year_1/project_work/Autoencoders/Results/VAE/2021_12_05_MNIST'
    print(models)
    print(config_file._conf)
    model_path = models.specify_folder(path=path)
    print(model_path)
    #
    if os.path.exists(join(model_path, 'model.pkl')):
        print("~~~~~~load model~~~~~~~")
        models = torch.load(join(model_path, 'model.pkl'))   
    else:
        print("~~~~~~train model~~~~~~~")
        logger = models.fit(data_1, val_set=True)     
        writer_legend = ResultsWriter(filepath = os.path.join(models.output_path, 'legend.txt'))
        writer_legend.write('Config file \n')
        writer_legend.write('%s\n'%config_file._conf)

    #models = GVCCA(input_dims=[784, 784], config=config_file._conf, classes=10).to(DEVICE)
    #logger = models.fit(target, data_1, data_2)

    MNIST_1 = datasets.MNIST('../data/MNIST', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    MNIST_2 = datasets.MNIST('../data/MNIST', train=False, download=True, transform=transforms.Compose([
           # transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation((180,180)),
            transforms.ToTensor()
        ]))
    data_1 = MNIST_1.test_data.reshape(-1, 784).float() / 255.
    target = MNIST_1.test_labels
    data_2 = MNIST_2.test_data.reshape(-1, 784).float() / 255.

   # latents = models.predict_latents(data_1,data_2)
    latents = models.predict_latents(data_1)
    models.plot_UMAP(data=latents, target=target, title=['Original MNIST latent space'], title_short='MNIST')
   # models.plot_UMAP(data=latents, target=target, title=['Original MNIST latent space', 'Rotated MNIST latent space'], title_short='MNIST')
    #models.print_reconstruction(data=latents, recon_type='test')