'''
Class for fitting and observing models


Optimisation_VAE: class for optimising VAE model

    preprocess(): function for getting data from generators

    generate_data(): function for creating generators from data

    optimise(): run optimisation

    optimise_batch(): function for passing data through model

    fit(): train model using training data

Optimisation_DVCCA: class for optimising DVCCA model which inherits methods from Optimisation_VAE

'''
from utils.datasets import MyDataset
import numpy as np
import torch
from utils.io_utils import Logger
from plot.plotting import Plotting
import datetime
import os

class Optimisation_VAE(Plotting):

    def __init__(self):
        super().__init__()

    def generate_data(self, data):
        generators = []

        for data_ in data:
            if self._config['mini_batch']:
                batch_sz = self._config['batch_size']
            else:
                batch_sz = np.shape(data_)[0]
            data_ = MyDataset(data_)
            kwargs_generator = {'num_workers': 4, 'pin_memory': True}
            generator = torch.utils.data.DataLoader(
                data_,
                batch_size=batch_sz,
                shuffle=False,
                **kwargs_generator
            )
            generators.append(generator)
        return generators

    def preprocess(self, generators, DEVICE):
        num_elements = len(generators[0].dataset)
        batch_size = generators[0].batch_size
        num_batches = len(generators[0])
        #torch.zeros(num_elements, generators[i].dataset.shape[1])
        data_out = [[] for i in range(len(generators))]
        data_1 = []
        data_2 = []
        for batch_idx, (data) in enumerate(zip(*generators)):
            data = [data_.to(DEVICE) for data_ in data]
            data_1.append(data[0])
            data_2.append(data[1])
            if self._config['mini_batch']:
                start = batch_idx * batch_size
                end = start + batch_size
                if batch_idx == num_batches - 1:
                    end = num_elements
                for i, data_ in enumerate(data):
                    #data_out[i].append(data_)
                    data_out[i].append(data_)
            else:
                data_out = data
        if self._config['mini_batch']:
            print(len(data_out))
            print(np.shape(data_out[0]))
            test = [[data_1], [data_2]]
            print(np.shape(test))
            #data_out = [[data_out[i]] for i in range(len(data_out))]
        print(np.shape(data_out))
        return data_out

    def init_optimisation(self):
        self.train()
    
    def end_optimisation(self):
        self.eval()

    def optimise(self, data):
        self.init_optimisation()
        self.minibatch = self._config['mini_batch']
        self.epochs = self._config['n_epochs']

        for epoch in range(1, self.epochs + 1):
            if self.minibatch:
                for batch_idx, local_batch in enumerate(data):
                    loss = self.optimise_batch(local_batch)
                    if batch_idx  == 0:
                        to_print = 'Train Epoch: ' + str(epoch) + ' ' + 'Train batch: ' + str(batch_idx) + ' '+ ', '.join([k + ': ' + str(round(v.item(), 3)) for k, v in loss.items()])
                        print(to_print)
            else:
                loss = self.optimise_batch(data)
                to_print = 'Train Epoch: ' + str(epoch) + ' ' + ', '.join([k + ': ' + str(round(v.item(), 3)) for k, v in loss.items()])
                print(to_print)
            if epoch == 1:
                log_keys = list(loss.keys())
                logger = Logger()
                logger.on_train_init(log_keys)
            else:
                logger.on_step_fi(loss)
        
        return logger

    def optimise_batch(self, local_batch):
        fwd_return = self.forward(local_batch)
        loss = self.loss_function(local_batch, fwd_return)
        [optimizer.zero_grad() for optimizer in self.optimizers]
        loss['total'].backward()
        [optimizer.step() for optimizer in self.optimizers]
        return loss
    
    def fit(self, *data):
        torch.manual_seed(42)  
        torch.cuda.manual_seed(42)
        DEVICE = torch.device("cuda")
        generators = self.generate_data(data)
        data = self.preprocess(generators, DEVICE)
        logger = self.optimise(data)
        if self._config['save_model']:
            self.out_path = self.format_folder(self)
            model_path = os.path.join(self.out_path, 'model.pkl')
            while os.path.exists(model_path):
                print("CAUTION! Model path already exists!")
                option = str(input("To overwrite type CONTINUE else type a suffix: "))
                if option == 'CONTINUE':
                    break
                else:
                    model_path = os.path.join(self.out_path, option + '.pkl')
            torch.save(self, model_path)
        self.plot_losses(logger)
        self.end_optimisation()
            
        return logger

    @staticmethod
    def format_folder(self):
        init_path = self._config['output_dir']
        model_type = self.model_type
        date = str(datetime.date.today())
        date = date.replace('-', '_')
        out_path = init_path + '/' + model_type + '/' + date
        #check if path exists
        while os.path.exists(os.path.join(os.getcwd(), out_path)):
            print("CAUTION! Path already exists!")
            option = str(input("To continue anyway type CONTINUE else type a suffix: "))
            if option == 'CONTINUE':
                break
            else:
                out_path = out_path + option
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.output_path = out_path    
        return out_path

    #TEST - not used yet
    def read_data(self, generator, prediction_size):
        num_batches = len(generator)
        num_elements = len(generator.dataset)
        batch_size = generator.batch_size
        dataset = torch.zeros(num_elements, prediction_size)
        for i, data in enumerate(generator):
            data = data.to(DEVICE)
            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements
            dataset[start:end] = data.cpu()
        return dataset.detach().numpy()
    
    def predict_latents(self, *data):
        DEVICE = torch.device("cuda")
        generators =  self.generate_data(data)
        data = self.preprocess(generators, DEVICE)
        num_elements = len(generators[0].dataset)
        batch_size = generators[0].batch_size
        num_batches = len(generators[0])
        with torch.no_grad():
            if self.minibatch:
                predictions = [torch.zeros(num_elements, int(self._config['latent_size'])) for i in range(len(generators))]
                for batch_idx, local_batch in enumerate(data):
                    mu, logvar = self.encode(local_batch)
                    pred = self.reparameterise(mu, logvar)
                    start = batch_idx * batch_size
                    end = start + batch_size
                    if batch_idx == num_batches - 1:
                        end = num_elements
                    for i, pred_ in enumerate(pred):
                        predictions[i][start:end, :] = pred_
            else:
                mu, logvar = self.encode(data)
                predictions = self.reparameterise(mu, logvar)

        return [predictions_.cpu().detach().numpy() for predictions_ in predictions]

    def predict_reconstruction(self, *data):
        DEVICE = torch.device("cuda")
        generators =  self.generate_data(data)
        data = self.preprocess(generators, DEVICE)
        print(np.shape(data[0][0]))
        print(np.shape(data[1][0]))
        print(np.shape(data))
        batch_size = generators[0].batch_size
        num_elements = len(generators[0].dataset)
        num_batches = len(generators[0])     
        with torch.no_grad():
            if self.minibatch:
                x_same = []
                x_cross = []
                for i in range(len(generators)):
                    x_same.append(torch.zeros(generators[i].dataset.shape[0], generators[i].dataset.shape[1]))
                    x_cross.append(torch.zeros(generators[i].dataset.shape[0], generators[i].dataset.shape[1]))
                for batch_idx, (local_batch) in enumerate(data):
                    print(np.shape(local_batch[0]))
                    print(np.shape(local_batch[1]))
                    exit()
                    mu, logvar = self.encode(local_batch)
                    z = self.reparameterise(mu, logvar)
                    x_same_, x_cross_ = self.decode(z)
                    start = batch_idx * batch_size
                    end = start + batch_size
                    if batch_idx == num_batches - 1:
                        end = num_elements
                    for i in range(len(generators)):
                        x_same[i][start:end, :] = x_same_[i]
                        x_cross[i][start:end, :] = x_cross_[i]
            else:
                mu, logvar = self.encode(data)
                z = self.reparameterise(mu, logvar)
                x_same, x_cross = self.decode(z)
        return [x_same_.cpu().detach().numpy() for x_same_ in x_same], [x_cross_.cpu().detach().numpy() for x_cross_ in x_cross]

class Optimisation_DVCCA(Optimisation_VAE):
    
    def __init__(self):
        super().__init__()

    def predict_reconstruction(self, *data):
        DEVICE = torch.device("cuda")
        generators =  self.generate_data(data)
        data = self.preprocess(generators, DEVICE)
        num_elements = len(generators[0].dataset)
        batch_size = generators[0].batch_size
        num_batches = len(generators[0])
        with torch.no_grad():
            if self.minibatch:
                for batch_idx, local_batch in enumerate(data):
                    mu, logvar = self.encode(local_batch)
                    z = self.reparameterise(mu, logvar)
                    x_recon = self.decode(z)
                    #NEED TO FINISH THIS SECTION
            else:
                mu, logvar = self.encode(data)
                z = self.reparameterise(mu, logvar)
                x_recon = self.decode(z)
        return [x_recon_.cpu().detach().numpy() for x_recon_ in x_recon]
    