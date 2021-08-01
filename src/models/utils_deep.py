from ..utils.datasets import MyDataset, MyDataset_SNPs, MyDataset_labels
import numpy as np
import torch
from torchvision import transforms
from ..utils.io_utils import Logger
from ..plot.plotting import Plotting
import datetime
import os
from sklearn.model_selection import KFold

class Optimisation_VAE(Plotting):
    def __init__(self):

        super().__init__()

    def generate_data(self, data, labels=None):
        batch_sz = self._config['batch_size']
        if self.SNP_model: 
            generators = [] 
            for data_ in data:
                data = MyDataset_SNPs(data)
                generator = torch.utils.data.DataLoader(
                    data,
                    batch_size=batch_sz,
                    shuffle=False,
                    **self.kwargs_generator
                )
                generators.append(generator)

            return generators
        elif labels is not None:
            generators = []
            for data_ in data:
                if self._config['batch_size'] is not None:
                    batch_sz = self._config['batch_size']
                else:
                    batch_sz = np.shape(data_)[0]
                data_ = MyDataset_labels(data_, labels)
                
                generator = torch.utils.data.DataLoader(
                    data_,
                    batch_size=batch_sz,
                    shuffle=False,
                    **self.kwargs_generator
                )
                generators.append(generator)

            return generators
        else: 
            generators = []

            for data_ in data:
                if batch_sz is not None:
                    batch_sz_tmp = self._config['batch_size']
                else:
                    batch_sz_tmp = np.shape(data_)[0]
    
                data_ = MyDataset(data_)
                generator = torch.utils.data.DataLoader(
                    data_,
                    batch_size=batch_sz_tmp,
                    shuffle=False,
                    **self.kwargs_generator
                )
                generators.append(generator)
                del generator
        
            return generators


    def preprocess(self, generators, labels=None):
        if labels is not None:   
            for batch_idx, batch in enumerate(zip(*generators)):
                data = [data_[0].to(self.device) for data_ in batch]
                labels = [data_[1].to(self.device, dtype=torch.int64) for data_ in batch]

            return [data, labels[0]]
        else:
            for batch_idx, data in enumerate(zip(*generators)):
                data = [data_.to(self.device) for data_ in data]
            return data

    def init_optimisation(self):
        self.train()
    
    def end_optimisation(self):
        self.eval()

    def centre_SNPs(self, data, MAF):
        MAF = torch.from_numpy(MAF).float()
        data = data - 2*MAF
        MAF = MAF.repeat(data.shape[0], 1)
        data = data - 2*MAF
        data[torch.isnan(data)] = 0 
        return data

    def optimise(self, generators=None, data=[], verbose=True):
        self.init_optimisation()
        self.to(self.device)
        self.epochs = self._config['n_epochs']
        self.transform = transform=transforms.Normalize(0, 1)
        for epoch in range(1, self.epochs + 1):
            if self.batch_size is not None and generators is not None:
                for batch_idx, (local_batch) in enumerate(zip(*generators)):
                    if self.SNP_model:
                        local_batch = [self.centre_SNPs(local_batch_, self.MAF_file) for local_batch_ in local_batch]  
                    if self.labels is not None:
                        labels = [local_batch_[1].to(self.device, dtype=torch.int64) for local_batch_ in local_batch]
                        local_batch = [local_batch_[0].to(self.device) for local_batch_ in local_batch] 
                        labels = labels[0]
                        local_batch = [local_batch, labels]
                    else:
                        local_batch = [local_batch_.to(self.device) for local_batch_ in local_batch]

                    loss = self.optimise_batch(local_batch)
                    if batch_idx  == 0 and verbose:
                        to_print = 'Train Epoch: ' + str(epoch) + ' ' + 'Train batch: ' + str(batch_idx) + ' '+ ', '.join([k + ': ' + str(round(v.item(), 3)) for k, v in loss.items()])
                        print(to_print)
            else:
                loss = self.optimise_batch(data)
                if verbose:
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

    def MSE_results(self, *data): 
        cross_recon = 0
        same_recon = 0
        with torch.no_grad():
            prediction = self.predict_reconstruction(*data)
            if self.joint_representation:       
                for i in range(self.n_views):
                    
                    same_temp = np.mean((prediction[i] - data[i])**2)
                    same_recon+= same_temp
                    data_input = list(data)
                    data_input[i] = np.zeros(np.shape(data[i]))
                    
                    pred_cross = self.predict_reconstruction(*data_input)
                    cross_temp = np.mean((pred_cross[i] - data[i])**2)
                    cross_recon+= cross_temp                       
            else:
                for i in range(self.n_views):
                    for j in range(self.n_views):
                        if i==j:
                            same_temp = np.mean((prediction[i][j] - data[i])**2)
                            same_recon+= same_temp
                        else:
                            cross_temp = np.mean((prediction[i][j] - data[i])**2)
                            cross_recon+= cross_temp
            return same_recon/self.n_views, cross_recon/self.n_views        

    def fit(self, *data, labels=None, MAF_file=None):
        self.batch_size = self._config['batch_size']
        self.data = data
        self.labels = labels
        self.MAF_file = MAF_file
        torch.manual_seed(42)  
        torch.cuda.manual_seed(42)
        use_GPU = self._config['use_GPU']
        use_cuda = use_GPU and torch.cuda.is_available()
        self.kwargs_generator = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.eps = 1e-15 

        generators = self.generate_data(self.data, labels=self.labels)
        if self.batch_size is not None:
            logger = self.optimise(generators=generators)
        else:
            if self.labels is not None:
                data = self.preprocess(generators, labels=True)
            else:
                data = self.preprocess(generators)
            logger = self.optimise(data=data)
        self.end_optimisation()
        if self._config['save_model']:
            model_path = os.path.join(self.output_path, 'model.pkl')
            while os.path.exists(model_path):
                print("CAUTION! Model path already exists!")
                option = str(input("To overwrite type CONTINUE else type a suffix: "))
                if option == 'CONTINUE':
                    break
                else:
                    model_path = os.path.join(self.output_path, option + '.pkl')
            torch.save(self, model_path)
            self.plot_losses(logger)
            
        return logger

    def specify_folder(self, path=None):
        if path is None:
            self.output_path = self.format_folder()
        else:
            self.output_path = path
        return self.output_path

    def format_folder(self):
        init_path = self._config['output_dir']
        model_type = self.model_type
        date = str(datetime.date.today())
        date = date.replace('-', '_')
        output_path = init_path + '/' + model_type + '/' + date
        if not os.path.exists(output_path):
            os.makedirs(output_path)   
        return output_path

    def predict_latents(self, *data, labels=None):
        self.data = data
        self.labels = labels
        generators =  self.generate_data(self.data, labels=self.labels)
        if self.batch_size is None:
            if self.labels is not None:
                data = self.preprocess(generators, labels=True)
            else:
                data = self.preprocess(generators)
        num_elements = len(generators[0].dataset)
        batch_size = generators[0].batch_size

        num_batches = len(generators[0])
        with torch.no_grad():
            if self.batch_size is not None:
                if self.joint_representation:
                    predictions = torch.zeros(num_elements, int(self._config['latent_size']))
                else:
                    predictions = [torch.zeros(num_elements, int(self._config['latent_size'])) for i in range(len(generators))]
                for batch_idx, local_batch in enumerate(zip(*generators)):
                    if self.labels is not None:
                        labels = [local_batch_[1].to(self.device, dtype=torch.int64) for local_batch_ in local_batch]
                        local_batch = [local_batch_[0].to(self.device) for local_batch_ in local_batch] 
                        labels = labels[0]
                        local_batch = [local_batch, labels]
                    else:
                        local_batch = [local_batch_.to(self.device) for local_batch_ in local_batch]
                    mu, logvar = self.encode(local_batch)
                    pred = self.reparameterise(mu, logvar)
                    if self.sparse:
                        pred = self.apply_threshold(pred)
                    start = batch_idx * batch_size
                    end = start + batch_size
                    if batch_idx == num_batches - 1:
                        end = num_elements
                    if self.joint_representation:
                        predictions[start:end, :] = pred
                    else:
                        for i, pred_ in enumerate(pred):
                            predictions[i][start:end, :] = pred_
            else:
                mu, logvar = self.encode(data)
                predictions = self.reparameterise(mu, logvar)
                if self.sparse:
                    predictions = self.apply_threshold(predictions)
        if self.joint_representation:
            return predictions.cpu().detach().numpy()
        return [predictions_.cpu().detach().numpy() for predictions_ in predictions]

    def predict_reconstruction(self, *data):
        generators =  self.generate_data(data)
        if self.batch_size==None:
            data = self.preprocess(generators)
        batch_size = generators[0].batch_size
        num_elements = len(generators[0].dataset)
        num_batches = len(generators[0])     
        with torch.no_grad():
            if self.batch_size!=None:
                x_recon_out = []
                if self.joint_representation:
                    for i in range(self.n_views):
                        x_recon_out.append(np.zeros((generators[i].dataset.shape[0], generators[i].dataset.shape[1])))
                else:
                    for i in range(self.n_views):
                        x_recon_temp = [np.zeros((generator.dataset.shape[0], generator.dataset.shape[1])) for generator in generators]
                        x_recon_out.append(x_recon_temp)

                for batch_idx, (local_batch) in enumerate(zip(*generators)):
                    local_batch = [local_batch_.to(self.device) for local_batch_ in local_batch]
                    mu, logvar = self.encode(local_batch)
                    z = self.reparameterise(mu, logvar)
                    if self.sparse:
                        z = self.apply_threshold(z)
                        
                    start = batch_idx * batch_size
                    end = start + batch_size
                    if batch_idx == num_batches - 1:
                        end = num_elements
                    x_recon = self.decode(z)                  
                    if self.joint_representation:
                        for i in range(self.n_views):
                            x_recon_out[i][start:end, :] = (self.sample_from_normal(x_recon[i])).cpu().detach().numpy()
                    else:
                        for i in range(self.n_views):
                            for j in range(self.n_views):
                                x_recon_out[i][j][start:end, :] = (self.sample_from_normal(x_recon[i][j])).cpu().detach().numpy()
            else:
                mu, logvar = self.encode(data)
                z = self.reparameterise(mu, logvar)
                if self.sparse:
                    z = self.apply_threshold(z)
                if self.joint_representation:
                    x_recon = self.decode(z)
                    x_recon_out = [(self.sample_from_normal(x_recon_)).cpu().detach().numpy() for x_recon_ in x_recon]
                else:
                    x_recon = self.decode(z)
                    x_recon_out = []
                    for i in range(self.n_views):
                        x_recon_temp = [(self.sample_from_normal(x_recon_)).cpu().detach().numpy() for x_recon_ in x_recon[i]]
                        x_recon_out.append(x_recon_temp)
            return x_recon_out

class Optimisation_AAE(Optimisation_VAE):
    
    def __init__(self):
        super().__init__()

    def optimise_batch(self, local_batch):
        self.init_optimisation()
        fwd_return = self.forward_recon(local_batch)
        loss_recon = self.recon_loss(self, local_batch, fwd_return)
        [optimizer.zero_grad() for optimizer in self.encoder_optimizers]
        [optimizer.zero_grad() for optimizer in self.decoder_optimizers]
        loss_recon.backward()
        [optimizer.step() for optimizer in self.encoder_optimizers]
        [optimizer.step() for optimizer in self.decoder_optimizers]

        fwd_return = self.forward_discrim(local_batch)
        loss_disc = self.discriminator_loss(self, fwd_return)
        self.discriminator_optimizer.zero_grad() 
        loss_disc.backward()
        self.discriminator_optimizer.step() 
        if self.wasserstein:
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
        fwd_return = self.forward_gen(local_batch)
        loss_gen = self.generator_loss(self, fwd_return) 
        [optimizer.zero_grad() for optimizer in self.generator_optimizers]
        loss_gen.backward()
        [optimizer.step() for optimizer in self.generator_optimizers]

        loss = {'recon': loss_recon,
                'disc': loss_disc,
                'gen': loss_gen}
        return loss

    def predict_reconstruction(self, *data):
        generators =  self.generate_data(data)
        if self.batch_size==None:
            data = self.preprocess(generators)
        num_elements = len(generators[0].dataset)
        batch_size = generators[0].batch_size
        num_batches = len(generators[0])
        with torch.no_grad():
            if self.batch_size!=None:
                x_same = []
                x_cross = []
                for batch_idx, local_batch in enumerate(zip(*generators)):
                    local_batch = [local_batch_.to(self.device) for local_batch_ in local_batch]
                    z = self.encode(local_batch)
                    if self.joint_representation:
                        x_same_ = self.decode(z)
                    else:
                        x_same_, x_cross_ = self.decode(z)
                    start = batch_idx * batch_size
                    end = start + batch_size
                    if batch_idx == num_batches - 1:
                        end = num_elements
                    for i in range(len(generators)):
                        if self.joint_representation:
                            x_same[i][start:end, :] = x_same_[i]
                        else:
                            x_same[i][start:end, :] = x_same_[i]
                            x_cross[i][start:end, :] = x_cross_[i]
            else:
                z = self.encode(data)
                if self.joint_representation:
                    x_same = self.decode(z)
                else:
                    x_same, x_cross = self.decode(z)
        if self.joint_representation:
            return [x_same_.cpu().detach().numpy() for x_same_ in x_same]
        return [x_same_.cpu().detach().numpy() for x_same_ in x_same], [x_cross_.cpu().detach().numpy() for x_cross_ in x_cross]

    def MSE_results(self, *data): 
        cross_recon = 0
        same_recon = 0
        with torch.no_grad():
            if self.joint_representation:
                x_same = self.predict_reconstruction(*data)
            else:
                x_same, x_cross = self.predict_reconstruction(*data)
            if self.joint_representation:  
                for i in range(self.n_views):
                    #same view prediction
                    same_temp = np.mean((x_same[i] - data[i])**2)
                    same_recon+= same_temp
                    data_input = list(data)
                    data_input[i] = np.zeros(np.shape(data[i]))
                    #cross view prediction
                    pred_cross = self.predict_reconstruction(*data_input)
                    cross_temp = np.mean((pred_cross[i] - data[i])**2)
                    cross_recon+= cross_temp  
            else:
                for i in range(self.n_views):
                    same_temp = np.mean((x_same[i] - data[i])**2)
                    same_recon+= same_temp
                    cross_temp = np.mean((x_cross[i] - data[i])**2)
                    cross_recon+= cross_temp
            return same_recon/self.n_views, cross_recon/self.n_views 

    def predict_latents(self, *data):
        generators =  self.generate_data(data)
        if self.batch_size==None:
            data = self.preprocess(generators)
        num_elements = len(generators[0].dataset)
        batch_size = generators[0].batch_size
        num_batches = len(generators[0])
        with torch.no_grad():
            if self.batch_size!=None:
                predictions = [torch.zeros(num_elements, int(self._config['latent_size'])) for i in range(len(generators))]
                for batch_idx, local_batch in enumerate(zip(*generators)):
                    local_batch = [local_batch_.to(self.device) for local_batch_ in local_batch]
                    pred = self.encode(local_batch)
                    start = batch_idx * batch_size
                    end = start + batch_size
                    if batch_idx == num_batches - 1:
                        end = num_elements
                    for i, pred_ in enumerate(pred):
                        predictions[i][start:end, :] = pred_
            else:
                predictions = self.encode(data)

        return [predictions_.cpu().detach().numpy() for predictions_ in predictions]


class Optimisation_DVCCA(Optimisation_VAE):
    
    def __init__(self):
        super().__init__()

    def predict_reconstruction(self, *data):
        generators =  self.generate_data(data)
        if self.batch_size==None:
            data = self.preprocess(generators)
        num_elements = len(generators[0].dataset)
        batch_size = generators[0].batch_size
        num_batches = len(generators[0])
        with torch.no_grad():
            if self.batch_size!=None:
                x_recon = []
                for i in range(len(generators)):
                    x_recon.append(torch.zeros(generators[i].dataset.shape[0], generators[i].dataset.shape[1]))
                for batch_idx, local_batch in enumerate(zip(*generators)):
                    local_batch = [local_batch_.to(self.device) for local_batch_ in local_batch]
                    mu, logvar = self.encode(local_batch)
                    z = self.reparameterise(mu, logvar)
                    x_recon_ = self.decode(z)
                    start = batch_idx * batch_size
                    end = start + batch_size
                    if batch_idx == num_batches - 1:
                        end = num_elements
                    for i in range(len(generators)):
                        x_recon[i][start:end, :] = x_recon_[i]
            else:
                mu, logvar = self.encode(data)
                z = self.reparameterise(mu, logvar)
                x_recon = self.decode(z)
        return [x_recon_.cpu().detach().numpy() for x_recon_ in x_recon]



class Optimisation_AE(Optimisation_VAE):
    
    def __init__(self):
        super().__init__()

    def predict_latents(self, *data):
        generators =  self.generate_data(data)
        if self.batch_size==None:
            data = self.preprocess(generators)
        num_elements = len(generators[0].dataset)
        batch_size = generators[0].batch_size
        num_batches = len(generators[0])
        with torch.no_grad():
            if self.batch_size!=None:
                predictions = [torch.zeros(num_elements, int(self._config['latent_size'])) for i in range(len(generators))]
                for batch_idx, local_batch in enumerate(zip(*generators)):
                    local_batch = [local_batch_.to(self.device) for local_batch_ in local_batch]
                    pred = self.encode(local_batch)
                    start = batch_idx * batch_size
                    end = start + batch_size
                    if batch_idx == num_batches - 1:
                        end = num_elements
                    for i, pred_ in enumerate(pred):
                        predictions[i][start:end, :] = pred_
            else:
                predictions = self.encode(data)

        return [predictions_.cpu().detach().numpy() for predictions_ in predictions]

    def predict_reconstruction(self, *data):
        generators =  self.generate_data(data)
        if self.batch_size==None:
            data = self.preprocess(generators)
        num_elements = len(generators[0].dataset)
        batch_size = generators[0].batch_size
        num_batches = len(generators[0])
        with torch.no_grad():
            if self.batch_size!=None:
                x_same = []
                x_cross = []
                for batch_idx, local_batch in enumerate(zip(*generators)):
                    local_batch = [local_batch_.to(self.device) for local_batch_ in local_batch]
                    z = self.encode(local_batch)
                    x_same_, x_cross_ = self.decode(z)
                    start = batch_idx * batch_size
                    end = start + batch_size
                    if batch_idx == num_batches - 1:
                        end = num_elements
                    for i in range(len(generators)):
                        x_same[i][start:end, :] = x_same_[i]
                        x_cross[i][start:end, :] = x_cross_[i]
            else:
                z = self.encode(data)
                x_same, x_cross = self.decode(z)
        return [x_same_.cpu().detach().numpy() for x_same_ in x_same], [x_cross_.cpu().detach().numpy() for x_cross_ in x_cross]

class Optimisation_GVCCA(Optimisation_VAE):
    
    def __init__(self):
        super().__init__()

    def fit(self, labels, *data):
        self.batch_size = self._config['batch_size']
        torch.manual_seed(42)  
        torch.cuda.manual_seed(42)
        use_GPU = self._config['use_GPU']
        use_cuda = use_GPU and torch.cuda.is_available()
        self.kwargs_generator = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        self.device = torch.device("cuda" if use_cuda else "cpu")

        generators = self.generate_data(data)
        if self.batch_size!=None:
            logger = self.optimise(generators, labels)
        else:
            data = self.preprocess(generators)
            logger = self.optimise(generators, labels, data)
        if self._config['save_model']:
            self.output_path = self.format_folder()
            model_path = os.path.join(self.output_path, 'model.pkl')
            while os.path.exists(model_path):
                print("CAUTION! Model path already exists!")
                option = str(input("To overwrite type CONTINUE else type a suffix: "))
                if option == 'CONTINUE':
                    break
                else:
                    model_path = os.path.join(self.output_path, option + '.pkl')
            torch.save(self, model_path)
            self.plot_losses(logger)
        self.end_optimisation()
            
        return logger

    def optimise(self, generators, labels, *data):
        self.init_optimisation()
        self.to(self.device)
        self.epochs = self._config['n_epochs']
        num_elements = len(generators[0].dataset)
        batch_size = generators[0].batch_size
        num_batches = len(generators[0])
        for epoch in range(1, self.epochs + 1):
            if self.batch_size!=None:
                for batch_idx, (local_batch) in enumerate(zip(*generators)):
                    local_batch = [local_batch_.to(self.device) for local_batch_ in local_batch]
                    start = batch_idx * batch_size
                    end = start + batch_size
                    if batch_idx == num_batches - 1:
                        end = num_elements
                    batch_labels = labels[start:end].to(self.device)

                    loss = self.optimise_batch(batch_labels, local_batch)
                    if batch_idx  == 0:
                        to_print = 'Train Epoch: ' + str(epoch) + ' ' + 'Train batch: ' + str(batch_idx) + ' '+ ', '.join([k + ': ' + str(round(v.item(), 3)) for k, v in loss.items()])
                        print(to_print)
            else:
                labels.to(self.device)
                loss = self.optimise_batch(labels, *data)
                to_print = 'Train Epoch: ' + str(epoch) + ' ' + ', '.join([k + ': ' + str(round(v.item(), 3)) for k, v in loss.items()])
                print(to_print)
            if epoch == 1:
                log_keys = list(loss.keys())
                logger = Logger()
                logger.on_train_init(log_keys)
            else:
                logger.on_step_fi(loss)
        
        return logger

    def optimise_batch(self, labels, *local_batch):
        fwd_return = self.forward(*local_batch)
        loss = self.loss_function(labels, fwd_return)
        [optimizer.zero_grad() for optimizer in self.optimizers]
        loss['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        [optimizer.step() for optimizer in self.optimizers]
        return loss