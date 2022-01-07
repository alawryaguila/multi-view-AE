from torch.utils.data.dataloader import DataLoader
from ..utils.dataloaders import MultiviewDataModule
import numpy as np
import torch
from torchvision import transforms
from ..utils.io_utils import Logger
from ..utils.trainer import trainer
from ..plot.plotting import Plotting
import datetime
import os
from sklearn.model_selection import KFold
import random 
from torchvision import datasets, transforms

class Optimisation_VAE(Plotting):
    
    def __init__(self):
        super().__init__() 

    def fit(self, *data, **kwargs):
        self.data = data
        self.labels = None
        self.val_set = False
        self.output_path = os.getcwd() #TODO - allow no path 
        self.eps = 1e-15 
        self.__dict__.update(kwargs)
        if not hasattr(self, 'trainer_dict') or not self.trainer_dict:
            self.trainer_dict = {'early_stopping': self.val_set}
        if not hasattr(self, 'batch_size') or not self.batch_size:
            self.batch_size = data[0].shape[0] if (type(data)==list or type(data)==tuple) else data.shape[0]

        torch.manual_seed(42)  
        torch.cuda.manual_seed(42)
        
        trainer_args = dict(output_path=self.output_path,
                        n_epochs=self.n_epochs,
                        **self.trainer_dict)

        #create trainer function
        py_trainer = trainer(**trainer_args)
        datamodule = MultiviewDataModule(data, batch_size=self.batch_size, val=self.val_set) #TO DO - create for other data formats
        py_trainer.fit(self, datamodule)

    def specify_folder(self, path=None):
        if path is None:
            self.output_path = self.format_folder()
        else:
            self.output_path = path
        return self.output_path

    def format_folder(self, output_dir='./'):
        init_path = output_dir
        model_type = self.model_type
        date = str(datetime.date.today())
        date = date.replace('-', '_')
        output_path = init_path + '/' + model_type + '/' + date
        if not os.path.exists(output_path):
            os.makedirs(output_path)   
        return output_path

    def predict_latents(self, *data, labels=None, val_set=False):
        self.labels = labels
        self.val_set = val_set
        #generators =  MultiviewDataModule.dataset(self.data, labels=self.labels) #TODO get working with labels
        dataset =  MultiviewDataModule.dataset(data)
        generator = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for batch_idx, local_batch in enumerate(generator): 
                local_batch = [local_batch_.to(self.device) for local_batch_ in local_batch]
                if self.variational:
                    mu, logvar = self.encode(local_batch)
                    pred = self.reparameterise(mu, logvar)
                else:
                    pred = self.encode(local_batch)
                if self.sparse:
                    pred = self.apply_threshold(pred)
                if batch_idx == 0:
                    predictions = self.process_output(pred, data_type='latent')
                else:
                    predictions = self.process_output(pred, pred=predictions, data_type='latent')
        return predictions

    def process_output(self, data, pred=None, data_type=None):
        if pred:
            if self.variational and data_type is None and self.dist=='gaussian':
                return [self.process_output(data_, pred=pred_, data_type=data_type) if isinstance(data_, list) else np.append(pred_, self.sample_from_normal(data_), axis=0) for pred_, data_ in zip(pred, data)]
            return [self.process_output(data_, pred=pred_, data_type=data_type) if isinstance(data_, list) else np.append(pred_,data_, axis=0) for pred_, data_ in zip(pred, data)]
        else:
            if self.variational and data_type is None and self.dist=='gaussian':
                return [self.process_output(data_, data_type=data_type) if isinstance(data_, list) else self.sample_from_normal(data_).cpu().detach().numpy() for data_ in data]
            return [self.process_output(data_, data_type=data_type) if isinstance(data_, list) else data_.cpu().detach().numpy() for data_ in data] #is cpu needed?

    def predict_reconstruction(self, *data):
        dataset =  MultiviewDataModule.dataset(data)
        generator = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)   
        with torch.no_grad():
            for batch_idx, (local_batch) in enumerate(generator):
                local_batch = [local_batch_.to(self.device) for local_batch_ in local_batch]
                if self.variational:
                    mu, logvar = self.encode(local_batch)
                    z = self.reparameterise(mu, logvar)
                else:
                    z = self.encode(local_batch)
                if self.sparse:
                    z = self.apply_threshold(z)
                x_recon = self.decode(z)   
                if batch_idx == 0:
                    x_reconstruction = self.process_output(x_recon)
                else:

                    x_reconstruction = self.process_output(x_recon, pred=x_reconstruction)
            return x_reconstruction

class Optimisation_AAE(Optimisation_VAE):
    
    def __init__(self):
        super().__init__()

    def validate_batch(self, local_batch):
        with torch.no_grad():
            self.eval()
            fwd_return = self.forward_recon(local_batch)
            loss_recon = self.recon_loss(self, local_batch, fwd_return)
            fwd_return = self.forward_discrim(local_batch)
            loss_disc = self.discriminator_loss(self, fwd_return)
            fwd_return = self.forward_gen(local_batch)
            loss_gen = self.generator_loss(self, fwd_return) 
            loss_total = loss_recon + loss_disc + loss_gen
            loss = {'total': loss_total,
                'recon': loss_recon,
                'disc': loss_disc,
                'gen': loss_gen}
        return loss

    def optimise_batch(self, local_batch):
        fwd_return = self.forward_recon(local_batch)
        loss_recon = self.recon_loss(self, local_batch, fwd_return)
        opts = self.optimizers()
        enc_opt = [opts.pop(0) for idx in range(self.n_views)]
        dec_opt = [opts.pop(0) for idx in range(self.n_views)]
        gen_opt = [opts.pop(0) for idx in range(self.n_views)]
        disc_opt = opts[0]
        [optimizer.zero_grad() for optimizer in enc_opt]
        [optimizer.zero_grad() for optimizer in dec_opt]
        self.manual_backward(loss_recon)
        [optimizer.step() for optimizer in enc_opt]
        [optimizer.step() for optimizer in dec_opt]

        fwd_return = self.forward_discrim(local_batch)
        loss_disc = self.discriminator_loss(self, fwd_return)
        disc_opt.zero_grad() 
        self.manual_backward(loss_disc)
        disc_opt.step() 
        if self.wasserstein:
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
        fwd_return = self.forward_gen(local_batch)
        loss_gen = self.generator_loss(self, fwd_return) 
        [optimizer.zero_grad() for optimizer in gen_opt]
        self.manual_backward(loss_gen)
        [optimizer.step() for optimizer in gen_opt]
        loss_total = loss_recon + loss_disc + loss_gen
        loss = {'total': loss_total,
                'recon': loss_recon,
                'disc': loss_disc,
                'gen': loss_gen}
        return loss